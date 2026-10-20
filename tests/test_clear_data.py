import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import server


class ClearDataTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.database = patch.object(server, 'DB', Path(self.folder.name) / 'test.sqlite3')
        self.database.start()
        server.init()
        server.worker = None
        server.state.update(running=False, segments=2, claims=1)
        server.store('segments', {'at': server.utc(), 'offset_s': 12, 'text': 'transcript'})
        server.store('claims', {'at': server.utc(), 'offset_s': 12, 'text': 'claim',
                     'status': 'aguardando pesquisa', 'explanation': 'test',
                     'sources': '[{"url":"https://example.org/evidence"}]', 'evidence_count': 1})
        with server.conn() as database:
            database.execute('INSERT INTO brave_search_attempts(attempted_at) VALUES (?)', (time.time(),))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, raise_app_exceptions=False),
                                        base_url='http://test')

    async def asyncTearDown(self):
        await server.stop()
        await self.client.aclose()
        self.database.stop()
        self.folder.cleanup()

    async def assert_data_preserved(self):
        feed = (await self.client.get('/api/feed')).json()
        self.assertEqual(len(feed['segments']), 1)
        self.assertEqual(len(feed['claims']), 1)
        self.assertEqual(server.brave_quota()['used'], 1)

    async def test_confirmed_clear_removes_history_but_preserves_brave_counter(self):
        response = await self.client.post('/api/clear', json={'confirmed': True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'cleared': True})
        self.assertEqual((await self.client.get('/api/feed')).json(), {'segments': [], 'paragraphs': [], 'claims': []})
        status = (await self.client.get('/api/status')).json()
        self.assertEqual(status['segments'], 0)
        self.assertEqual(status['claims'], 0)
        self.assertEqual(status['brave_quota']['used'], 1)
        self.assertTrue(status['can_clear'])
        server.init()
        self.assertEqual(server.brave_quota()['used'], 1)

    async def test_explicit_boolean_confirmation_is_required(self):
        for body, expected in [({}, 422), ({'confirmed': False}, 400), ({'confirmed': 'true'}, 422)]:
            with self.subTest(body=body):
                response = await self.client.post('/api/clear', json=body)
                self.assertEqual(response.status_code, expected)
                await self.assert_data_preserved()

    async def test_running_or_stopping_capture_blocks_clear_on_server(self):
        server.state['running'] = True
        self.assertFalse((await self.client.get('/api/status')).json()['can_clear'])
        self.assertEqual((await self.client.post('/api/clear', json={'confirmed': True})).status_code, 409)
        await self.assert_data_preserved()
        server.state['running'] = False
        server.worker = asyncio.create_task(asyncio.Event().wait())
        self.assertFalse((await self.client.get('/api/status')).json()['can_clear'])
        self.assertEqual((await self.client.post('/api/clear', json={'confirmed': True})).status_code, 409)
        await self.assert_data_preserved()

    async def test_failed_deletion_rolls_back_entire_history(self):
        with server.conn() as database:
            database.executescript("""CREATE TRIGGER block_clear BEFORE DELETE ON segments
                                  BEGIN SELECT RAISE(ABORT, 'test transaction rollback'); END;""")
        self.assertEqual((await self.client.post('/api/clear', json={'confirmed': True})).status_code, 500)
        await self.assert_data_preserved()
        self.assertEqual(server.state['segments'], 2)
        self.assertEqual(server.state['claims'], 1)


if __name__ == '__main__':
    unittest.main()
