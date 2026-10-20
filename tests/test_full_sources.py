import asyncio
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import full_sources
import server


QUOTE = 'O Observatório Alfa registrou 20 eventos em 2025.'
TEXT = QUOTE + ' O relatório apresenta a metodologia utilizada na contagem dos eventos. ' * 5
HTML = ('<html><head><title>Relatório Alfa</title><meta property="article:published_time" '
        'content="2025-06-12T10:00:00Z"></head><body><nav>Entrar</nav><article><h1>Relatório Alfa</h1>'
        '<p>' + TEXT + '</p><p>Os responsáveis descrevem critérios de inclusão e exclusão, '
        'aplicados ao longo do período de observação. A documentação reúne registros '
        'e contextualiza os dados apresentados no relatório anual.</p></article></body></html>')


def document(source_id=0):
    return {'source_id': source_id, 'url': 'https://example.org/report', 'title': 'Relatório Alfa',
            'published_date': '2025-06-12', 'fetched_at': '2026-10-20T12:00:00Z',
            'status': 'ok', 'text': TEXT, 'error': None}


def assessment(verdict='fato'):
    return {'verdict': verdict, 'explanation': 'Evidência no artigo.', 'citations': [
        {'source_id': 0, 'quote': QUOTE, 'relation': 'supports' if verdict == 'fato' else 'contradicts'}]}


class SourceReaderTests(unittest.TestCase):
    def test_extractor_gets_article_and_date_from_real_html(self):
        result = full_sources.extract_document(HTML.encode('utf-8'), 'https://example.org/report')
        self.assertIn(QUOTE, result['text'])
        self.assertEqual(result['published_date'], '2025-06-12')
        self.assertNotIn('Entrar', result['text'])

    def test_private_addresses_and_non_https_targets_are_rejected(self):
        for url in ('http://example.org', 'file:///etc/passwd', 'https://user:password@example.org', 'https://example.org:444'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                full_sources.public_target(url)
        for address in ('127.0.0.1', '10.0.0.1', '169.254.169.254', '::1', 'fd00::1', '224.0.0.1'):
            with self.subTest(address=address), patch('full_sources.socket.getaddrinfo', return_value=[
                    (socket.AF_INET, socket.SOCK_STREAM, 0, '', (address, 443))]), self.assertRaises(ValueError):
                full_sources.public_target('https://example.org')

    def test_connection_uses_pinned_ip_and_original_tls_hostname(self):
        connection = full_sources.PublicHTTPSConnection('example.org', '93.184.216.34')
        connection._context = Mock()
        with patch('full_sources.socket.create_connection') as connect:
            connection.connect()
        connect.assert_called_once_with(('93.184.216.34', 443), timeout=10)
        self.assertEqual(connection._context.wrap_socket.call_args.kwargs['server_hostname'], 'example.org')

    def test_redirect_to_private_host_is_blocked_before_connecting(self):
        response = Mock(status=302)
        response.getheader.return_value = 'https://127.0.0.1/private'
        connection = Mock()
        connection.getresponse.return_value = response
        def target(url):
            if '127.0.0.1' in url:
                raise ValueError('private target')
            return full_sources.urlsplit(url), '93.184.216.34'
        with patch('full_sources.public_target', side_effect=target), \
             patch('full_sources.PublicHTTPSConnection', return_value=connection) as create:
            with self.assertRaises(ValueError):
                full_sources.download_html('https://example.org/report')
        create.assert_called_once()
        connection.close.assert_called_once()

    def test_unsupported_and_oversized_responses_are_not_extracted(self):
        for content_type, length in [('application/pdf', '10'), ('text/html', str(full_sources.MAX_BYTES + 1))]:
            response = Mock(status=200)
            headers = {'Content-Type': content_type, 'Content-Length': length}
            response.getheader.side_effect = lambda key, default=None: headers.get(key, default)
            connection = Mock()
            connection.getresponse.return_value = response
            with patch('full_sources.public_target', return_value=(full_sources.urlsplit('https://example.org'), '93.184.216.34')), \
                 patch('full_sources.PublicHTTPSConnection', return_value=connection), self.assertRaises(ValueError):
                full_sources.download_html('https://example.org')
            response.read1.assert_not_called()

    def test_block_limit_is_explicit_and_no_block_exceeds_model_budget(self):
        text = ('Uma frase completa para o contexto do relatório. ' * 1000).strip()
        blocks, complete = full_sources.text_blocks(text)
        self.assertFalse(complete)
        self.assertEqual(len(blocks), full_sources.MAX_BLOCKS)
        self.assertTrue(all(len(block) <= full_sources.BLOCK_SIZE for block in blocks))


class FullAssessmentTests(unittest.IsolatedAsyncioTestCase):
    def model(self, outcomes):
        return Mock(path=Path('test.llamafile'), assess_claim=AsyncMock(side_effect=outcomes))

    async def test_full_text_is_sent_with_exact_quote_offsets(self):
        model = self.model([assessment()])
        doc = document()
        result = await full_sources.assess_documents(model, QUOTE, [doc])
        self.assertEqual(result['verdict'], 'fato')
        self.assertTrue(result['complete'])
        citation = result['citations'][0]
        self.assertEqual(doc['text'][citation['char_start']:citation['char_end']], citation['quote'])
        self.assertEqual(model.assess_claim.call_args.kwargs['basis'], 'full_text')

    async def test_unavailable_source_conflict_or_truncation_forces_inconclusive(self):
        unavailable = dict(document(1), status='unavailable', text='')
        result = await full_sources.assess_documents(self.model([assessment()]), QUOTE, [document(), unavailable])
        self.assertEqual(result['verdict'], 'inconclusivo')
        self.assertFalse(result['complete'])
        result = await full_sources.assess_documents(self.model([assessment(), assessment('fake')]), QUOTE, [document(), document(1)])
        self.assertEqual(result['verdict'], 'inconclusivo')
        doc = dict(document(), text=TEXT * 100)
        with patch.object(full_sources, 'MAX_BLOCKS', 1):
            result = await full_sources.assess_documents(self.model([assessment()]), QUOTE, [doc])
        self.assertEqual(result['verdict'], 'inconclusivo')
        self.assertFalse(result['complete'])

    async def test_model_failure_preserves_source_but_does_not_classify(self):
        result = await full_sources.assess_documents(self.model([RuntimeError('test')]), QUOTE, [document()])
        self.assertEqual(result['verdict'], 'inconclusivo')
        self.assertFalse(result['complete'])


class FullSourceAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.database = patch.object(server, 'DB', Path(self.folder.name) / 'full.sqlite3')
        self.database.start()
        server.init()
        server.worker = None
        server.state['running'] = False
        server.manual_searches.clear()
        server.claim_analyses.clear()
        self.ready = patch.object(server.local_llm, 'status', return_value={'ready': True, 'state': 'pronta'})
        self.ready.start()
        self.search = patch.object(server, 'search_sources', side_effect=AssertionError('No new Brave call allowed'))
        self.search.start()
        server.store('claims', {'at': server.utc(), 'offset_s': 12, 'text': QUOTE, 'status': 'inconclusivo',
                     'explanation': 'Original search', 'sources': json.dumps([{'title': 'Source',
                     'url': 'https://example.org/report', 'snippet': 'Original snippet'}]), 'evidence_count': 1})
        with server.conn() as database:
            database.execute('INSERT INTO manual_claim_searches VALUES (?, ?)', (1, server.utc()))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test')

    async def asyncTearDown(self):
        await self.client.aclose()
        self.search.stop()
        self.ready.stop()
        self.database.stop()
        self.folder.cleanup()

    async def test_saves_article_and_citations_reuses_complete_analysis_and_preserves_quota(self):
        with patch.object(server, 'read_source', return_value=document()) as reader, \
             patch.object(server.local_llm, 'assess_claim', new=AsyncMock(return_value=assessment())):
            self.assertEqual((await self.client.post('/api/claims/1/analyze-full')).status_code, 200)
            self.assertTrue((await self.client.post('/api/claims/1/analyze-full')).json()['already_analyzed'])
            reader.assert_called_once()
        claim = (await self.client.get('/api/feed')).json()['claims'][0]
        self.assertEqual(claim['full_assessment']['verdict'], 'fato')
        self.assertEqual(claim['full_documents'][0]['published_date'], '2025-06-12')
        self.assertNotIn('text', claim['full_documents'][0])
        with server.conn() as database:
            saved = json.loads(database.execute('SELECT document FROM claim_full_sources').fetchone()[0])
            self.assertEqual(saved['text'], TEXT)
        self.assertEqual(server.brave_quota()['used'], 0)
        self.assertEqual((await self.client.post('/api/clear', json={'confirmed': True})).status_code, 200)
        with server.conn() as database:
            self.assertEqual(database.execute('SELECT COUNT(*) FROM claim_full_sources').fetchone()[0], 0)
            self.assertEqual(database.execute('SELECT COUNT(*) FROM claim_full_assessments').fetchone()[0], 0)

    async def test_full_read_blocks_parallel_requests_and_cleanup(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def assess(*args, **kwargs):
            entered.set()
            await release.wait()
            return assessment()
        with patch.object(server, 'read_source', return_value=document()), \
             patch.object(server.local_llm, 'assess_claim', new=assess):
            task = asyncio.create_task(self.client.post('/api/claims/1/analyze-full'))
            try:
                await asyncio.wait_for(entered.wait(), timeout=2)
                self.assertEqual((await self.client.post('/api/claims/1/analyze-full')).status_code, 409)
                self.assertEqual((await self.client.post('/api/clear', json={'confirmed': True})).status_code, 409)
            finally:
                release.set()
                await task
        self.assertFalse(server.full_reads)
        self.assertFalse(server.claim_analyses)


if __name__ == '__main__':
    unittest.main()
