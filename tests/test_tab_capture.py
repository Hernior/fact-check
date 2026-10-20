import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import server


class TabCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.database = patch.object(server, 'DB', Path(self.folder.name) / 'test.sqlite3')
        self.database.start()
        server.init()
        server.state.update(running=False)
        server.worker = None
        server.tab_queue = None
        self.revision = patch.object(server.local_llm, 'revise_transcript', new=AsyncMock(
            side_effect=lambda text, **kwargs: {'join_previous': False, 'text': text}))
        self.revision.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test')

    async def asyncTearDown(self):
        await server.stop()
        self.revision.stop()
        await self.client.aclose()
        self.database.stop()
        self.folder.cleanup()

    async def test_local_pipeline_stores_audio_offset_without_paid_services(self):
        with patch.object(server, 'transcribe', new=AsyncMock(return_value='Uma afirmação transcrita.')) as transcribe, \
             patch.object(server.local_llm, 'extract_claims', new=AsyncMock(return_value=['Uma afirmação transcrita.'])) as extract, \
             patch.object(server, 'openai_json', side_effect=AssertionError('Cloud AI invoked')), \
             patch.object(server, 'search_sources', side_effect=AssertionError('Search invoked')):
            response = await self.client.post('/api/tab/start')
            self.assertEqual(response.status_code, 200)
            session = response.json()['session']
            audio = b'\x00\x00' * 16000
            response = await self.client.post('/api/tab/audio', params={'session': session}, content=audio)
            self.assertEqual(response.status_code, 200)
            for _ in range(100):
                feed = (await self.client.get('/api/feed')).json()
                if feed['segments'] and feed['paragraphs']:
                    break
                await asyncio.sleep(0.01)
            await self.client.post('/api/tab/stop', params={'session': session})
            await server.stop()
            feed = (await self.client.get('/api/feed')).json()
            self.assertEqual(feed['segments'][0]['text'], 'Uma afirmação transcrita.')
            self.assertEqual(feed['segments'][0]['offset_s'], 1)
            self.assertEqual(feed['claims'][0]['text'], 'Uma afirmação transcrita.')
            self.assertEqual(feed['claims'][0]['status'], 'aguardando pesquisa')
            self.assertEqual(feed['claims'][0]['sources'], [])
            self.assertEqual(feed['claims'][0]['offset_s'], 1)
            extract.assert_awaited_once_with('Uma afirmação transcrita.', debate=True, context='', participants=[])
            transcribe.assert_awaited_once_with(audio)
            new_session = (await self.client.post('/api/tab/start')).json()['session']
            self.assertEqual((await self.client.post('/api/tab/start')).status_code, 409)
            self.assertEqual((await self.client.post('/api/start', json={'url': 'https://youtu.be/abcdefghijk'})).status_code, 409)
            self.assertNotEqual(new_session, session)
            self.assertEqual((await self.client.post('/api/tab/audio', params={'session': session}, content=audio)).status_code, 409)
            self.assertEqual((await self.client.post('/api/tab/stop', params={'session': session})).status_code, 409)
            self.assertTrue(server.state['running'])

    async def test_audio_validation_and_disk_backlog(self):
        gate = asyncio.Event()

        async def slow_transcribe(audio):
            await gate.wait()
            return ''

        with patch.object(server, 'transcribe', new=slow_transcribe):
            session = (await self.client.post('/api/tab/start')).json()['session']
            params = {'session': session}
            for audio in (b'', b'\x00'):
                self.assertEqual((await self.client.post('/api/tab/audio', params=params, content=audio)).status_code, 400)
            oversized = b'\x00' * (server.SAMPLE_RATE * 2 * server.CHUNK_SECONDS + 2)
            self.assertEqual((await self.client.post('/api/tab/audio', params=params, content=oversized)).status_code, 413)
            response = await self.client.post('/api/tab/audio', params=params, content=b'\x00\x00')
            self.assertEqual(response.status_code, 200)
            await asyncio.sleep(0.01)
            for _ in range(12):
                self.assertEqual((await self.client.post('/api/tab/audio', params=params, content=b'\x00\x00')).status_code, 200)
            self.assertEqual((await self.client.post('/api/tab/audio', params=params, content=b'\x00\x00')).status_code, 200)
            gate.set()

    async def test_worklet_and_dashboard_are_served(self):
        for path in ('/', '/audio-worklet.js', '/tab-capture.js'):
            self.assertEqual((await self.client.get(path)).status_code, 200)
        page = (await self.client.get('/')).text
        self.assertIn('Capturar áudio da aba', page)
        self.assertNotIn('Capturar ?udio', page)

    async def test_local_model_failure_preserves_transcription_and_next_chunk(self):
        with patch.object(server, 'transcribe', new=AsyncMock(return_value='Outro trecho transcrito.')), \
             patch.object(server.local_llm, 'extract_claims', new=AsyncMock(side_effect=[RuntimeError('model unavailable'), []])), \
             self.assertLogs('server', level='ERROR'):
            session = (await self.client.post('/api/tab/start')).json()['session']
            for _ in range(2):
                await self.client.post('/api/tab/audio', params={'session': session}, content=b'\x00\x00' * 16000)
            for _ in range(100):
                if server.state['segments'] == 2 and server.state['error']:
                    break
                await asyncio.sleep(0.01)
            await server.stop()
            self.assertEqual(server.state['segments'], 2)
            self.assertFalse(server.state['running'])
            self.assertIn('identificar', server.state['error'])
            self.assertEqual((await self.client.get('/api/feed')).json()['claims'], [])

    async def test_transcription_uses_cpu_without_cuda_and_consumes_segments(self):
        model = Mock()
        model.transcribe.return_value = (iter([SimpleNamespace(text=' Texto local. ')]), None)
        with patch.object(server, 'whisper', None), \
             patch('faster_whisper.WhisperModel', return_value=model) as create_model:
            text = await server.transcribe(b'\x00\x00' * 16000)
            self.assertEqual(text, 'Texto local.')
            create_model.assert_called_once_with(server.MODEL_SIZE, device='cpu', compute_type='int8')
            audio = model.transcribe.call_args.args[0]
            self.assertEqual(audio.shape, (16000,))
            self.assertEqual(str(audio.dtype), 'float32')


if __name__ == '__main__':
    unittest.main()
