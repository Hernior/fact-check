import asyncio
import json
import tempfile
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import server
from local_llm import LocalLlamafile


SOURCES = [{'title': 'Relatório Alfa', 'url': 'https://example.org/report',
            'snippet': 'O Observatório Alfa registrou 20 eventos em 2025.'}]


def suggestion(verdict='fato', relation='supports'):
    return {'verdict': verdict, 'explanation': 'O trecho apresenta a contagem para 2025.',
            'sufficient': True, 'conflicting': False,
            'citations': [{'source_id': 0, 'quote': SOURCES[0]['snippet'], 'relation': relation}]}


class AssessmentGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_grounded_support_and_contradiction_can_produce_suggestions(self):
        model = LocalLlamafile(Path.cwd())
        for verdict, relation in [('fato', 'supports'), ('fake', 'contradicts')]:
            with self.subTest(verdict=verdict), patch.object(model, '_complete', new=AsyncMock(return_value=suggestion(verdict, relation))):
                result = await model.assess_claim('Claim', SOURCES)
                self.assertEqual(result['verdict'], verdict)
                self.assertEqual(result['citations'][0]['quote'], SOURCES[0]['snippet'])

    async def test_missing_evidence_is_inconclusive_without_model_request(self):
        model = LocalLlamafile(Path.cwd())
        with patch.object(model, '_complete', new=AsyncMock()) as complete:
            self.assertEqual((await model.assess_claim('Claim', []))['verdict'], 'inconclusivo')
            complete.assert_not_awaited()

    async def test_unsafe_or_ungrounded_outputs_are_downgraded(self):
        model = LocalLlamafile(Path.cwd())
        cases = [dict(suggestion('fake', 'contradicts'), sufficient=False),
                 dict(suggestion(), conflicting=True), dict(suggestion(), sufficient='true'),
                 dict(suggestion(), citations=[]),
                 dict(suggestion(), citations=[{'source_id': 99, 'quote': SOURCES[0]['snippet'], 'relation': 'supports'}]),
                 dict(suggestion(), citations=[{'source_id': 0, 'quote': 'Uma citação inventada pelo modelo.', 'relation': 'supports'}]),
                 suggestion('fake', 'supports'), suggestion('fato', 'contradicts')]
        for candidate in cases:
            with self.subTest(candidate=candidate), patch.object(model, '_complete', new=AsyncMock(return_value=candidate)):
                self.assertEqual((await model.assess_claim('Claim', SOURCES))['verdict'], 'inconclusivo')

    async def test_missing_quantity_is_not_fake_even_if_model_claims_contradiction(self):
        model = LocalLlamafile(Path.cwd())
        snippet = 'O Observatório Alfa publicou seu relatório anual, sem divulgar quantidades.'
        candidate = suggestion('fake', 'contradicts')
        candidate['citations'][0]['quote'] = snippet
        candidate['explanation'] = 'A ausência de informação sobre a quantidade contradiz a afirmação.'
        with patch.object(model, '_complete', new=AsyncMock(return_value=candidate)):
            self.assertEqual((await model.assess_claim('O Observatório Alfa registrou 20 eventos em 2025.',
                             [dict(SOURCES[0], snippet=snippet)]))['verdict'], 'inconclusivo')

    async def test_support_missing_claim_numbers_cannot_be_labeled_fact(self):
        model = LocalLlamafile(Path.cwd())
        with patch.object(model, '_complete', new=AsyncMock(return_value=suggestion())):
            self.assertEqual((await model.assess_claim('O Observatório Alfa registrou 30 eventos em 2025.',
                                                     SOURCES))['verdict'], 'inconclusivo')


class AssessmentAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.database = patch.object(server, 'DB', Path(self.folder.name) / 'assessment.sqlite3')
        self.database.start()
        server.init()
        server.state['running'] = False
        server.worker = None
        server.claim_analyses.clear()
        server.analysis_errors.clear()
        self.ready = patch.object(server.local_llm, 'status', return_value={'ready': True, 'state': 'pronta'})
        self.ready.start()
        self.search = patch.object(server, 'search_sources', side_effect=AssertionError('Brave must not be called'))
        self.search.start()
        self.cloud = patch.object(server, 'openai_json', side_effect=AssertionError('Cloud AI must not be called'))
        self.cloud.start()
        server.store('claims', {'at': server.utc(), 'offset_s': 12, 'text': 'O Observatório Alfa registrou 20 eventos em 2025.',
                     'status': 'inconclusivo', 'explanation': 'Snippet-only research.',
                     'sources': json.dumps(SOURCES), 'evidence_count': 1})
        with server.conn() as database:
            database.execute('INSERT INTO manual_claim_searches VALUES (?, ?)', (1, server.utc()))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test')

    async def asyncTearDown(self):
        await self.client.aclose()
        self.cloud.stop()
        self.search.stop()
        self.ready.stop()
        self.database.stop()
        self.folder.cleanup()

    async def test_assessment_is_saved_separately_without_spending_quota(self):
        result = {'verdict': 'fato', 'explanation': 'Corroboração preliminar.', 'citations': suggestion()['citations']}
        with patch.object(server.local_llm, 'assess_claim', new=AsyncMock(return_value=result)) as assess:
            self.assertEqual((await self.client.post('/api/claims/1/analyze')).status_code, 200)
            self.assertTrue((await self.client.post('/api/claims/1/analyze')).json()['already_analyzed'])
            assess.assert_awaited_once()
        claim = (await self.client.get('/api/feed')).json()['claims'][0]
        self.assertEqual(claim['status'], 'inconclusivo')  # suggestion is not an editorial verdict
        self.assertEqual(claim['assessment']['verdict'], 'fato')
        self.assertEqual(claim['assessment']['citations'][0]['source_id'], 0)
        self.assertTrue(claim['assessment']['assessed_at'])
        self.assertEqual(server.brave_quota()['used'], 0)

    async def test_analysis_blocks_duplicate_request_and_clear_until_done(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def assess(*args):
            entered.set()
            await release.wait()
            return {'verdict': 'inconclusivo', 'explanation': 'Insuficiente.', 'citations': []}
        with patch.object(server.local_llm, 'assess_claim', new=assess):
            task = asyncio.create_task(self.client.post('/api/claims/1/analyze'))
            try:
                await asyncio.wait_for(entered.wait(), timeout=2)
                self.assertEqual((await self.client.post('/api/claims/1/analyze')).status_code, 409)
                self.assertEqual((await self.client.post('/api/clear', json={'confirmed': True})).status_code, 409)
            finally:
                release.set()
                await task
        self.assertEqual((await self.client.post('/api/clear', json={'confirmed': True})).status_code, 200)
        with server.conn() as database:
            self.assertEqual(database.execute('SELECT COUNT(*) FROM claim_assessments').fetchone()[0], 0)

    async def test_missing_search_and_model_failure_do_not_trigger_new_research(self):
        with server.conn() as database:
            database.execute('DELETE FROM manual_claim_searches')
        self.assertEqual((await self.client.post('/api/claims/1/analyze')).status_code, 409)
        with server.conn() as database:
            database.execute('INSERT INTO manual_claim_searches VALUES (?, ?)', (1, server.utc()))
        with patch.object(server.local_llm, 'assess_claim', new=AsyncMock(side_effect=RuntimeError('test'))), self.assertLogs('server', level='ERROR'):
            self.assertEqual((await self.client.post('/api/claims/1/analyze')).status_code, 502)
        claim = (await self.client.get('/api/feed')).json()['claims'][0]
        self.assertIsNone(claim['assessment'])
        self.assertTrue(claim['analysis_error'])
        self.assertEqual(claim['sources'], SOURCES)
        self.assertFalse(server.claim_analyses)

    async def test_manual_search_automatically_analyzes_saved_results(self):
        with server.conn() as database:
            database.execute('DELETE FROM manual_claim_searches')
        result = {'verdict': 'fato', 'explanation': 'Sugestão com referência.', 'citations': suggestion()['citations']}
        with patch.dict(os.environ, {'BRAVE_SEARCH_API_KEY': 'test-only'}), \
             patch.object(server, 'search_sources', new=AsyncMock(return_value=SOURCES)) as search, \
             patch.object(server.local_llm, 'assess_claim', new=AsyncMock(return_value=result)) as assess:
            self.assertEqual((await self.client.post('/api/claims/1/search')).status_code, 200)
            search.assert_awaited_once()
            assess.assert_awaited_once()
        claim = (await self.client.get('/api/feed')).json()['claims'][0]
        self.assertEqual(claim['assessment']['verdict'], 'fato')
        self.assertEqual(claim['sources'], SOURCES)


if __name__ == '__main__':
    unittest.main()
