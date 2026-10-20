import asyncio
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import server
from local_llm import LocalLlamafile


class RevisionTests(unittest.IsolatedAsyncioTestCase):
    def test_long_paragraph_is_analyzed_in_overlapping_passages(self):
        self.assertEqual(server.claim_passages('Primeira. Segunda. Terceira. Quarta.'),
                         ['Primeira. Segunda. Terceira.', 'Terceira. Quarta.'])

    async def test_joined_text_with_inconsistent_flag_is_recognized(self):
        model = LocalLlamafile('.')
        with patch.object(model, '_complete', new=AsyncMock(return_value={
            'join_previous': False, 'text': 'A cidade tem vinte escolas.'})):
            result = await model.revise_transcript('vinte escolas.', previous='A cidade tem')
        self.assertTrue(result['join_previous'])

    async def test_revision_can_correct_words_using_context_and_audio_hypothesis(self):
        model = LocalLlamafile('.')
        result = {'join_previous': False, 'text': 'Eu tive 56 milhões de votos.'}
        with patch.object(model, '_complete', new=AsyncMock(return_value=result)) as complete:
            revised = await model.revise_transcript('Eu tive 56 milhões de voos.',
                        context='A conversa é sobre eleições.', audio_transcript=result['text'])
        self.assertEqual(revised, result)
        self.assertIn('eleições', complete.call_args.args[0][-1]['content'])
        self.assertIn('votos', complete.call_args.args[0][-1]['content'])

    async def test_invalid_rewrites_and_changed_numbers_are_rejected(self):
        model = LocalLlamafile('.')
        for result in [
            {'join_previous': False, 'text': 'O Brasil tem 27 estados.'},
            {'join_previous': False, 'text': 'Os oceanos estão quentes.'},
            {'join_previous': True, 'text': 'O Brasil tem 26 estados.'},
            {'join_previous': 'false', 'text': 'O Brasil tem 26 estados.'},
            {'join_previous': False, 'text': ''},
        ]:
            with self.subTest(result=result), patch.object(model, '_complete', new=AsyncMock(return_value=result)):
                with self.assertRaises(RuntimeError):
                    await model.revise_transcript('O Brasil tem 26 estados.')


class ParagraphPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.database = patch.object(server, 'DB', Path(self.folder.name) / 'test.sqlite3')
        self.database.start()
        server.init()
        server.worker = None
        server.tab_queue = None
        server.state.update(running=False)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test')
        self.queue = asyncio.Queue()

    async def asyncTearDown(self):
        await server.stop()
        await self.client.aclose()
        self.database.stop()
        self.folder.cleanup()

    async def run_chunks(self, chunks, revisions, claims=None):
        async def transcribe(audio):
            # Distinct bytes identify each raw chunk; joined audio is a second hypothesis.
            return chunks[audio[0] - 1] if len(audio) == 32000 else 'Áudio conjunto contextual.'

        with patch.object(server, 'transcribe', new=AsyncMock(side_effect=transcribe)) as whisper, \
             patch.object(server.local_llm, 'revise_transcript', new=AsyncMock(side_effect=revisions)) as revise, \
             patch.object(server.local_llm, 'extract_claims', new=AsyncMock(return_value=claims or [])) as extract, \
             patch.object(server, 'search_sources', side_effect=AssertionError('Brave invoked')), \
             patch.object(server, 'openai_json', side_effect=AssertionError('Cloud invoked')):
            server.state.update(running=True, segments=0, claims=0, error=None)
            server.worker = asyncio.create_task(server.run_tab(self.queue))
            for index in range(len(chunks)):
                await self.queue.put(bytes([index + 1, 0]) * 16000)
            for _ in range(200):
                if server.state['segments'] == len(chunks) and self.queue.empty() and revise.await_count >= len(chunks):
                    break
                await asyncio.sleep(0.005)
            await server.stop()
            return (await self.client.get('/api/feed')).json(), whisper, revise, extract

    async def test_join_then_topic_change_extracts_completed_text_and_keeps_originals(self):
        full = 'O Brasil tem 26 estados e um Distrito Federal.'
        feed, whisper, revise, extract = await self.run_chunks([
            'O Brasil tem 26 estados e um', 'Distrito Federal.', 'Recife é a capital de Pernambuco.'
        ], [
            {'join_previous': False, 'text': 'O Brasil tem 26 estados e um'},
            {'join_previous': True, 'text': full},
            {'join_previous': False, 'text': 'Recife é a capital de Pernambuco.'}
        ], claims=[full])
        self.assertEqual(len(feed['segments']), 3)
        self.assertEqual(len(feed['paragraphs']), 2)
        paragraph = feed['paragraphs'][1]
        self.assertEqual(paragraph['text'], full)
        self.assertEqual([o['text'] for o in paragraph['originals']], ['O Brasil tem 26 estados e um', 'Distrito Federal.'])
        self.assertEqual((paragraph['start_s'], paragraph['offset_s']), (0, 2))
        self.assertTrue(all(p['finalized'] for p in feed['paragraphs']))
        self.assertEqual(extract.await_args_list[0].args, (full,))
        self.assertEqual(extract.await_count, 2)
        self.assertEqual(len(feed['claims']), 1, 'Repeated claims within a capture must be deduplicated')
        self.assertEqual(whisper.await_count, 3)
        self.assertEqual(revise.await_args_list[1].kwargs['audio_transcript'], 'Distrito Federal.')

    async def test_three_chunk_bound_and_stop_flush(self):
        feed, _, _, extract = await self.run_chunks(['Primeiro', 'Segundo', 'Terceiro'], [
            {'join_previous': False, 'text': 'Primeiro'},
            {'join_previous': True, 'text': 'Primeiro Segundo'},
            {'join_previous': True, 'text': 'Primeiro Segundo Terceiro'}])
        self.assertEqual(len(feed['paragraphs']), 1)
        self.assertTrue(feed['paragraphs'][0]['finalized'])
        extract.assert_awaited_once_with('Primeiro Segundo Terceiro', debate=True, context='', participants=[])

    async def test_second_whisper_pass_only_after_invalid_revision_without_backlog(self):
        with patch.object(server.local_llm, 'status', return_value={'ready': True}):
            _, whisper, revise, _ = await self.run_chunks(['A cidade tem', 'vinte escolas.'], [
                {'join_previous': False, 'text': 'A cidade tem'}, RuntimeError('invalid revision'),
                {'join_previous': True, 'text': 'A cidade tem vinte escolas.'}])
        self.assertEqual(whisper.await_count, 3)
        self.assertEqual(revise.await_count, 3)
        self.assertEqual(revise.await_args_list[-1].kwargs['audio_transcript'], 'Áudio conjunto contextual.')

    async def test_second_whisper_pass_is_skipped_when_more_audio_is_waiting(self):
        with patch.object(server.local_llm, 'status', return_value={'ready': True}), self.assertLogs('server', level='ERROR'):
            _, whisper, _, _ = await self.run_chunks(['Primeiro.', 'Segundo.', 'Terceiro.'], [
                {'join_previous': False, 'text': 'Primeiro.'}, RuntimeError('invalid revision'),
                {'join_previous': False, 'text': 'Terceiro.'}])
        self.assertEqual(whisper.await_count, 3)

    async def test_revision_failure_preserves_raw_and_continues(self):
        with self.assertLogs('server', level='ERROR'):
            feed, _, _, extract = await self.run_chunks(['Original.', 'Próximo.'], [
                RuntimeError('model unavailable'), {'join_previous': False, 'text': 'Próximo.'}])
        self.assertEqual(len(feed['paragraphs']), 2)
        self.assertEqual(feed['paragraphs'][1]['text'], 'Original.')
        self.assertIn('preservada', feed['paragraphs'][1]['revision_error'])
        self.assertEqual(extract.await_count, 2)

    async def test_stop_during_revision_keeps_pending_original_and_extracts_it(self):
        started = asyncio.Event()

        async def review(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        with patch.object(server, 'transcribe', new=AsyncMock(return_value='Texto original.')), \
             patch.object(server.local_llm, 'revise_transcript', new=review), \
             patch.object(server.local_llm, 'extract_claims', new=AsyncMock(return_value=[])) as extract:
            server.state.update(running=True)
            server.worker = asyncio.create_task(server.run_tab(self.queue))
            await self.queue.put(b'\x00\x00' * 16000)
            await asyncio.wait_for(started.wait(), 2)
            await server.stop()
            feed = (await self.client.get('/api/feed')).json()
        self.assertEqual(feed['paragraphs'][0]['text'], 'Texto original.')
        self.assertIn('pendente', feed['paragraphs'][0]['revision_error'])
        extract.assert_awaited_once_with('Texto original.', debate=True, context='', participants=[])

    async def test_clear_removes_revisions_but_keeps_quota(self):
        await self.run_chunks(['Texto.'], [{'join_previous': False, 'text': 'Texto.'}])
        server.brave_quota(reserve=True)
        self.assertEqual((await self.client.post('/api/clear', json={'confirmed': True})).status_code, 200)
        with closing(server.conn()) as database:
            self.assertEqual(database.execute('SELECT COUNT(*) FROM transcript_paragraphs').fetchone()[0], 0)
        self.assertEqual(server.brave_quota()['used'], 1)

    async def test_concurrent_stop_waits_for_final_extraction_and_blocks_clear(self):
        extracting, release = asyncio.Event(), asyncio.Event()

        async def extract(text, **kwargs):
            extracting.set()
            await release.wait()
            return []

        with patch.object(server, 'transcribe', new=AsyncMock(return_value='Texto.')), \
             patch.object(server.local_llm, 'revise_transcript', new=AsyncMock(return_value={
                 'join_previous': False, 'text': 'Texto.'})), \
             patch.object(server.local_llm, 'extract_claims', new=extract):
            server.state.update(running=True)
            server.worker = asyncio.create_task(server.run_tab(self.queue))
            await self.queue.put(b'\x00\x00' * 16000)
            while server.state['segments'] != 1:
                await asyncio.sleep(0.005)
            first = asyncio.create_task(server.stop())
            await asyncio.wait_for(extracting.wait(), 2)
            second = asyncio.create_task(server.stop())
            await asyncio.sleep(0.005)
            status = (await self.client.get('/api/status')).json()
            self.assertTrue(status['stopping'])
            self.assertFalse(status['can_clear'])
            self.assertEqual((await self.client.post('/api/clear', json={'confirmed': True})).status_code, 409)
            self.assertEqual((await self.client.post('/api/tab/start')).status_code, 409)
            release.set()
            await asyncio.gather(first, second)
        self.assertTrue((await self.client.get('/api/feed')).json()['paragraphs'][0]['finalized'])


if __name__ == '__main__':
    unittest.main()
