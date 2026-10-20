import asyncio
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import server
from capture_store import DiskCapture, new_session


class CaptureSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.db = patch.object(server, 'DB', Path(self.folder.name) / 'queue.sqlite3')
        self.db.start()
        server.init()
        server.worker = server.tab_queue = None
        server.state.update(running=False, source=None, segments=0, claims=0)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test')
        self.audio = b'\x01\x00' * 16000
        self.transcribe = patch.object(server, 'transcribe', new=AsyncMock(return_value='O Brasil tem 26 estados.'))
        self.revise = patch.object(server.local_llm, 'revise_transcript', new=AsyncMock(
            side_effect=lambda text, **kwargs: {'join_previous': False, 'text': text}))
        self.extract = patch.object(server.local_llm, 'extract_claims', new=AsyncMock(return_value=['O Brasil tem 26 estados.']))
        self.transcribe.start(); self.revise.start(); self.extract.start()

    async def asyncTearDown(self):
        await server.stop()
        self.extract.stop(); self.revise.stop(); self.transcribe.stop()
        await self.client.aclose()
        server.tab_queue = server.worker = None
        self.db.stop()
        self.folder.cleanup()

    async def start(self):
        response = await self.client.post('/api/tab/start')
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['session']

    async def send(self, session, sequence=None, audio=None):
        params = {'session': session}
        if sequence is not None:
            params['sequence'] = sequence
        return await self.client.post('/api/tab/audio', params=params, content=self.audio if audio is None else audio)

    async def wait_for(self, predicate):
        for _ in range(300):
            if predicate():
                return
            await asyncio.sleep(0.005)
        self.fail('Processing condition not reached')

    async def test_disk_backlog_survives_slow_whisper_and_stop_drains_every_received_block(self):
        gate = asyncio.Event()
        async def slow(audio):
            await gate.wait()
            return 'Texto.'
        try:
            with patch.object(server, 'transcribe', new=slow):
                session = await self.start()
                for sequence in range(12):
                    self.assertEqual((await self.send(session, sequence)).status_code, 200)
                status = (await self.client.get('/api/status')).json()
                self.assertEqual(status['progress']['received'], 12)
                self.assertEqual(status['progress']['transcribed_seconds'], 0)
                # Closing/reopening connections proves these are disk records, not RAM references.
                with closing(server.conn()) as db:
                    self.assertEqual(db.execute('SELECT SUM(length(pcm)) FROM capture_audio').fetchone()[0], len(self.audio) * 12)
                response = await self.client.post('/api/tab/stop', params={'session': session})
                self.assertTrue(response.json()['processing'])
                self.assertEqual((await self.send(session)).status_code, 409)
                self.assertEqual((await self.client.post('/api/tab/start')).status_code, 409)
                gate.set()
                await asyncio.wait_for(server.stop(), 4)
        finally:
            gate.set()
        progress = (await self.client.get('/api/status')).json()['progress']
        self.assertEqual(progress['analyzed_seconds'], 12)
        self.assertEqual(progress['pending'], 0)
        self.assertEqual(progress['failed'], 0)
        self.assertEqual(server.brave_quota()['used'], 0)

    async def test_text_analysis_does_not_block_whisper(self):
        gate, entered = asyncio.Event(), asyncio.Event()
        async def slow(*args, **kwargs):
            entered.set()
            await gate.wait()
            return {'join_previous': False, 'text': args[0]}
        try:
            with patch.object(server.local_llm, 'revise_transcript', new=slow):
                session = await self.start()
                for sequence in range(8):
                    await self.send(session, sequence)
                await asyncio.wait_for(entered.wait(), 2)
                await self.wait_for(lambda: server.tab_queue.progress()['transcribed_seconds'] == 8)
                self.assertTrue(server.state['running'])
                self.assertEqual(server.tab_queue.progress()['analyzed_seconds'], 0)
                gate.set()
                await server.stop()
        finally:
            gate.set()

    async def test_new_capture_archives_old_results_and_legacy_without_deleting_evidence(self):
        legacy = server.store('segments', {'at': server.utc(), 'offset_s': 1, 'text': 'Histórico antigo.'})
        first = await self.start()
        await self.send(first)
        await server.stop()
        saved = (await self.client.get('/api/feed')).json()
        claim_id = saved['claims'][0]['id']
        server.store('manual_claim_searches', {'claim_id': claim_id, 'searched_at': server.utc()})
        server.store('claim_assessments', {'claim_id': claim_id, 'assessed_at': server.utc(), 'verdict': 'inconclusivo',
            'explanation': 'Evidência preservada.', 'citations': '[]', 'model': 'fixture'})
        second = await self.start()
        current = (await self.client.get('/api/feed')).json()
        self.assertEqual(current, {'segments': [], 'paragraphs': [], 'claims': []})
        archived = (await self.client.get('/api/feed', params={'session_id': first})).json()
        self.assertEqual(archived['claims'][0]['assessment']['explanation'], 'Evidência preservada.')
        sessions = (await self.client.get('/api/sessions')).json()
        self.assertEqual(sessions[0]['id'], second)
        self.assertTrue(next(s for s in sessions if s['id'] == first)['archived'])
        old = (await self.client.get('/api/feed', params={'session_id': 'legacy'})).json()
        self.assertEqual(old['segments'][0]['id'], legacy)
        self.assertEqual((await self.client.get('/api/feed?session_id=unknown')).status_code, 404)

    async def test_audio_retries_are_idempotent_and_out_of_order_is_not_silently_accepted(self):
        session = await self.start()
        self.assertEqual((await self.send(session, 0)).status_code, 200)
        self.assertEqual((await self.send(session, 0)).status_code, 200)
        self.assertEqual((await self.send(session, 0, b'\x02\x00')).status_code, 409)
        self.assertEqual((await self.send(session, 2)).status_code, 409)
        self.assertEqual(server.tab_queue.progress()['received'], 1)

    async def test_disk_write_error_does_not_acknowledge_or_erase_audio(self):
        session = await self.start()
        await self.send(session)
        with patch.object(server.tab_queue, 'append', side_effect=sqlite3.OperationalError('disk full')), self.assertLogs('server', level='ERROR'):
            self.assertEqual((await self.send(session)).status_code, 507)
        self.assertEqual(server.tab_queue.progress()['received'], 1)

    async def test_restart_recovers_unfinished_jobs_and_does_not_reopen_browser_capture(self):
        new_session(server.conn, 'saved', server.utc(), 'tab', [])
        queue = DiskCapture(server.conn, 'saved')
        queue.append(self.audio, 0)
        # This session was still marked capturing when its process disappeared.
        await server.resume_captures()
        progress = queue.progress()
        self.assertEqual(progress['pending'], 0)
        self.assertEqual(progress['analyzed_seconds'], 1)
        self.assertFalse(server.state['running'])
        self.assertTrue(queue.closed())
        self.assertEqual((await self.client.get('/api/feed')).json()['claims'][0]['text'], 'O Brasil tem 26 estados.')

    async def test_restart_reuses_transcript_and_rebuilds_only_unfinished_analysis(self):
        new_session(server.conn, 'saved', server.utc(), 'tab', [])
        queue = DiskCapture(server.conn, 'saved')
        queue.append(self.audio)
        segment = server.store('segments', {'session_id': 'saved', 'at': server.utc(), 'offset_s': 1, 'text': 'O Brasil tem 26 estados.'})
        server.store('transcript_paragraphs', {'session_id': 'saved', 'text': 'Revisão interrompida.', 'originals': '[]', 'finalized': 0})
        with closing(server.conn()) as db, db:
            db.execute("UPDATE capture_audio SET status='analyzing',segment_id=?", (segment,))
        with patch.object(server, 'transcribe', side_effect=AssertionError('Already transcribed')):
            await server.resume_captures()
        feed = (await self.client.get('/api/feed')).json()
        self.assertEqual(len(feed['segments']), 1)
        self.assertEqual(len(feed['paragraphs']), 1)
        self.assertEqual(len(feed['claims']), 1)
        self.assertEqual(queue.progress()['analyzed_seconds'], 1)

    async def test_shutdown_preserves_in_flight_audio_for_next_process(self):
        entered = asyncio.Event()
        async def in_flight(audio):
            entered.set()
            await asyncio.Event().wait()
        with patch.object(server, 'transcribe', new=in_flight):
            session = await self.start()
            await self.send(session, 0)
            await self.send(session, 1)
            await asyncio.wait_for(entered.wait(), 2)
            queue = server.tab_queue
            queue.aborting = True  # same ordering as the application shutdown
            server.worker.cancel()
            await asyncio.gather(server.worker, return_exceptions=True)
        self.assertEqual(queue.progress()['received'], 2)
        self.assertEqual(queue.progress()['pending'], 2)
        server.init()
        await server.resume_captures()
        self.assertEqual(queue.progress()['analyzed_seconds'], 2)
        self.assertFalse(server.state['running'])

    async def test_confirmed_clear_deletes_archives_and_audio_but_preserves_quota(self):
        session = await self.start()
        await self.send(session)
        await server.stop()
        server.brave_quota(reserve=True)
        self.assertEqual((await self.client.post('/api/clear', json={'confirmed': True})).status_code, 200)
        with closing(server.conn()) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM capture_audio').fetchone()[0], 0)
        self.assertEqual((await self.client.get('/api/sessions')).json(), [])
        self.assertEqual(server.brave_quota()['used'], 1)


if __name__ == '__main__':
    unittest.main()
