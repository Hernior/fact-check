import asyncio
import os
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx
import server


class BraveQuotaTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.database = patch.object(server, 'DB', Path(self.folder.name) / 'quota.sqlite3')
        self.database.start()
        server.init()
        self.environment = patch.dict(os.environ, {'BRAVE_SEARCH_API_KEY': 'test-only-not-a-real-key'})
        self.environment.start()
        self.real_client = httpx.AsyncClient

    async def asyncTearDown(self):
        self.environment.stop()
        self.database.stop()
        self.folder.cleanup()

    def seed(self, count, at=None):
        with server.conn() as database:
            database.executemany('INSERT INTO brave_search_attempts(attempted_at) VALUES (?)',
                                 [(at if at is not None else time.time(),)] * count)

    def transport(self, handler):
        return patch.object(server.httpx, 'AsyncClient', side_effect=lambda **options:
                            self.real_client(transport=httpx.MockTransport(handler), **options))

    async def test_concurrent_searches_cannot_exceed_1000(self):
        self.seed(995)
        sent = []

        def respond(request):
            sent.append(request)
            # The reservation has to be committed before the network call.
            self.assertGreaterEqual(server.brave_quota()['used'], 996)
            return httpx.Response(200, json={'web': {'results': []}})

        with self.transport(respond):
            results = await asyncio.gather(*(server.search_sources('teste') for _ in range(20)),
                                           return_exceptions=True)
        self.assertEqual(len(sent), 5)
        self.assertEqual(sum(isinstance(result, server.BraveQuotaExceeded) for result in results), 15)
        self.assertEqual(server.brave_quota()['used'], 1000)
        self.assertEqual(server.brave_quota()['remaining'], 0)

    async def test_limit_survives_initialization_and_key_rotation(self):
        self.seed(1000)
        server.init()
        with patch.dict(os.environ, {'BRAVE_SEARCH_API_KEY': 'another-test-only-key'}), \
             self.transport(lambda _: self.fail('HTTP must not be called at the limit')):
            assessment = await server.verify_claim('teste')
        self.assertEqual(assessment['status'], 'sem evidências')
        self.assertIn('1.000', assessment['explanation'])
        self.assertEqual(server.brave_quota()['used'], 1000)

    async def test_errors_timeouts_and_cancellation_are_not_refunded_or_retried(self):
        cases = [(httpx.HTTPStatusError, lambda request: httpx.Response(500, json={})),
                 (httpx.ReadTimeout, lambda request: self.raise_error(httpx.ReadTimeout('test timeout'))),
                 (asyncio.CancelledError, lambda request: self.raise_error(asyncio.CancelledError()))]
        for index, (error, response) in enumerate(cases, start=1):
            with self.subTest(error=error.__name__), self.transport(response):
                with self.assertRaises(error):
                    await server.search_sources('teste')
            self.assertEqual(server.brave_quota()['used'], index)

    @staticmethod
    def raise_error(error):
        raise error

    async def test_missing_key_and_database_failure_do_not_send_http(self):
        with patch.dict(os.environ, {'BRAVE_SEARCH_API_KEY': ''}), \
             self.transport(lambda _: self.fail('No key: HTTP must not be called')):
            self.assertEqual(await server.search_sources('teste'), [])
        self.assertEqual(server.brave_quota()['used'], 0)
        with patch.object(server, 'conn', side_effect=sqlite3.OperationalError('test locked database')), \
             self.transport(lambda _: self.fail('DB failure: HTTP must not be called')):
            with self.assertRaises(sqlite3.OperationalError):
                await server.search_sources('teste')
        self.assertEqual(server.brave_quota()['used'], 0)

    async def test_month_change_does_not_prematurely_restore_quota(self):
        timestamp = datetime(2026, 10, 31, 23, 59, tzinfo=timezone.utc).timestamp()
        self.seed(1000, timestamp)
        with patch.object(server.time, 'time', return_value=timestamp + 120):
            self.assertTrue(server.brave_quota()['blocked'])
            with self.assertRaises(server.BraveQuotaExceeded):
                server.brave_quota(reserve=True)
        with patch.object(server.time, 'time', return_value=timestamp + 32 * 86400):
            quota = server.brave_quota(reserve=True)
            self.assertEqual(quota['used'], 1)
            self.assertEqual(quota['remaining'], 999)
        # Expired attempts remain in the audit trail.
        with server.conn() as database:
            self.assertEqual(database.execute('SELECT COUNT(*) FROM brave_search_attempts').fetchone()[0], 1001)

    async def test_status_exposes_only_counts_and_reading_does_not_spend_quota(self):
        self.seed(2)
        async with self.real_client(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
            for _ in range(2):
                response = await client.get('/api/status')
                self.assertEqual(response.status_code, 200)
                quota = response.json()['brave_quota']
                self.assertEqual(quota, {'limit': 1000, 'used': 2, 'remaining': 998,
                                         'window_days': 32, 'blocked': False})
                self.assertNotIn('test-only-not-a-real-key', response.text)


if __name__ == '__main__':
    unittest.main()
