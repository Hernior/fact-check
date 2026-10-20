import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import server


class ManualSearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.database = patch.object(server, 'DB', Path(self.folder.name) / 'search.sqlite3')
        self.database.start()
        self.environment = patch.dict(os.environ, {'BRAVE_SEARCH_API_KEY': 'fake-search-key'})
        self.environment.start()
        server.init()
        server.state['running'] = False
        server.worker = None
        server.manual_searches.clear()
        server.store('claims', {'at': server.utc(), 'offset_s': 12, 'text': 'Recife é a capital de Pernambuco.',
                     'status': 'aguardando pesquisa', 'explanation': 'Not researched',
                     'sources': '[]', 'evidence_count': 0})
        self.real_client = httpx.AsyncClient
        self.client = self.real_client(transport=httpx.ASGITransport(app=server.app), base_url='http://test')
        self.cloud = patch.object(server, 'openai_json', side_effect=AssertionError('Cloud AI must not be called'))
        self.cloud.start()
        self.sent = []

    async def asyncTearDown(self):
        await self.client.aclose()
        self.cloud.stop()
        self.environment.stop()
        self.database.stop()
        self.folder.cleanup()

    def transport(self, handler):
        def counted(request):
            self.sent.append(request)
            return handler(request)
        return patch.object(server.httpx, 'AsyncClient', side_effect=lambda **options:
                            self.real_client(transport=httpx.MockTransport(counted), **options))

    async def claim(self):
        return (await self.client.get('/api/feed')).json()['claims'][0]

    async def test_manual_search_saves_snippets_without_truth_verdict_and_reuses_result(self):
        def response(request):
            self.assertEqual(request.url.params['q'], 'Recife é a capital de Pernambuco.')
            self.assertEqual(server.brave_quota()['used'], 1)
            return httpx.Response(200, json={'web': {'results': [
                {'title': '<b>Fonte</b>', 'url': 'https://example.org/source', 'description': '<em>Trecho</em>'}]}})
        self.assertIsNone((await self.claim())['searched_at'])
        with self.transport(response):
            self.assertEqual((await self.client.post('/api/claims/1/search')).status_code, 200)
            server.init()  # completed searches survive restart/initialization
            self.assertTrue((await self.client.post('/api/claims/1/search')).json()['already_searched'])
        claim = await self.claim()
        self.assertEqual(claim['status'], 'inconclusivo')
        self.assertIn('fontes completas', claim['explanation'])
        self.assertEqual(claim['sources'][0]['snippet'], '<em>Trecho</em>')
        self.assertTrue(claim['searched_at'])
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(server.brave_quota()['used'], 1)

    async def test_empty_search_is_completed_without_repeated_quota_spending(self):
        with self.transport(lambda _: httpx.Response(200, json={'web': {'results': []}})):
            await self.client.post('/api/claims/1/search')
            await self.client.post('/api/claims/1/search')
        self.assertEqual((await self.claim())['status'], 'sem evidências')
        self.assertEqual(len(self.sent), 1)

    async def test_pending_search_blocks_duplicate_click_and_clear(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def response(request):
            entered.set()
            await release.wait()
            return httpx.Response(200, json={})
        with self.transport(response):
            first = asyncio.create_task(self.client.post('/api/claims/1/search'))
            try:
                await asyncio.wait_for(entered.wait(), timeout=2)
                self.assertTrue((await self.claim())['searching'])
                self.assertEqual((await self.client.post('/api/claims/1/search')).status_code, 409)
                self.assertFalse((await self.client.get('/api/status')).json()['can_clear'])
                self.assertEqual((await self.client.post('/api/clear', json={'confirmed': True})).status_code, 409)
            finally:
                release.set()
                await first
        self.assertEqual(len(self.sent), 1)
        self.assertFalse(server.manual_searches)
        self.assertEqual((await self.client.post('/api/clear', json={'confirmed': True})).status_code, 200)
        with server.conn() as database:
            self.assertEqual(database.execute('SELECT COUNT(*) FROM manual_claim_searches').fetchone()[0], 0)
        self.assertEqual(server.brave_quota()['used'], 1)

    async def test_missing_key_unknown_claim_and_long_query_never_send_request(self):
        with self.transport(lambda _: self.fail('HTTP must not be sent')):
            self.assertEqual((await self.client.post('/api/claims/999/search')).status_code, 404)
            with patch.dict(os.environ, {'BRAVE_SEARCH_API_KEY': ''}):
                self.assertEqual((await self.client.post('/api/claims/1/search')).status_code, 503)
            with server.conn() as database:
                database.execute('UPDATE claims SET text = ? WHERE id = 1', ('a' * 601,))
            self.assertEqual((await self.client.post('/api/claims/1/search')).status_code, 422)
        self.assertEqual(server.brave_quota()['used'], 0)

    async def test_exhausted_quota_blocks_http_and_keeps_claim_unchanged(self):
        with server.conn() as database:
            database.executemany('INSERT INTO brave_search_attempts(attempted_at) VALUES (?)', [(time.time(),)] * 1000)
        with self.transport(lambda _: self.fail('Quota exhausted: no HTTP allowed')):
            self.assertEqual((await self.client.post('/api/claims/1/search')).status_code, 429)
        self.assertEqual((await self.claim())['status'], 'aguardando pesquisa')
        self.assertIsNone((await self.claim())['searched_at'])
        self.assertEqual(server.brave_quota()['used'], 1000)

    async def test_failure_counts_once_and_allows_only_explicit_retry(self):
        with self.transport(lambda _: httpx.Response(500)), self.assertLogs('server', level='ERROR'):
            self.assertEqual((await self.client.post('/api/claims/1/search')).status_code, 502)
        self.assertEqual(server.brave_quota()['used'], 1)
        self.assertIsNone((await self.claim())['searched_at'])
        self.assertFalse(server.manual_searches)
        with self.transport(lambda _: httpx.Response(200, json={})):
            self.assertEqual((await self.client.post('/api/claims/1/search')).status_code, 200)
        self.assertEqual(server.brave_quota()['used'], 2)


if __name__ == '__main__':
    unittest.main()
