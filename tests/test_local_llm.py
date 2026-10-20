import asyncio
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

import server
from local_llm import LocalLlamafile


def completion(content):
    return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                           'message': {'content': json.dumps(content)}}]})


def approved_review(subject):
    return completion({'decisions': [{'id': 0, 'kind': 'fact', 'self_contained': True,
                                      'externally_verifiable': True, 'subject': subject}]})


class LocalLlamafileTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        database = patch.object(server, 'DB', Path(self.folder.name) / 'test.sqlite3')
        database.start()
        self.addCleanup(database.stop)
        server.init()
        self.model = LocalLlamafile(Path(self.folder.name))
        self.real_client = httpx.AsyncClient

    def transport(self, handler):
        return patch('local_llm.httpx.AsyncClient', side_effect=lambda **options:
                     self.real_client(transport=httpx.MockTransport(handler), **options))

    def make_ready(self):
        self.model.ready = True
        self.model.process = Mock()
        self.model.process.poll.return_value = None

    async def test_owned_process_waits_for_health_then_is_terminated(self):
        executable = Path(self.folder.name) / 'fake.llamafile.exe'
        executable.touch()
        self.model.path = executable
        self.model.port = 0
        process = Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        probes = []

        def response(request):
            probes.append(request.url.path)
            if request.url.path == '/health':
                return httpx.Response(503 if probes.count('/health') == 1 else 200, json={'status': 'ok'})
            self.assertEqual(request.headers['Authorization'], 'Bearer ' + self.model.token)
            return httpx.Response(200, json={'data': [{'id': 'Qwen-local'}]})

        with patch('local_llm.subprocess.Popen', return_value=process) as spawn, self.transport(response):
            await self.model.start()
            self.assertTrue(self.model.status()['ready'])
            self.assertEqual(probes, ['/health', '/v1/models', '/health', '/v1/models'])
            options = spawn.call_args.kwargs
            self.assertNotIn('OPENAI_API_KEY', options['env'])
            self.assertNotIn('BRAVE_SEARCH_API_KEY', options['env'])
            self.assertIn('--server', spawn.call_args.args[0])
            self.assertIn('127.0.0.1', spawn.call_args.args[0])
            self.assertIn('disable', spawn.call_args.args[0])
            await self.model.stop()
        process.terminate.assert_called_once()
        self.assertIsNone(self.model.process)
        self.assertIsNone(self.model.log)
        self.assertFalse(self.model.ready)

    async def test_missing_binary_does_not_launch_any_process(self):
        self.model.path = Path(self.folder.name) / 'missing.llamafile'
        with patch('local_llm.subprocess.Popen') as spawn, self.assertLogs('local_llm', level='ERROR'):
            await self.model.start()
        spawn.assert_not_called()
        self.assertFalse(self.model.ready)
        self.assertEqual(self.model.phase, 'indisponível')

    async def test_timeout_cleans_up_only_the_owned_process(self):
        executable = Path(self.folder.name) / 'fake.exe'
        executable.touch()
        self.model.path = executable
        self.model.port = 0
        self.model.startup_timeout = 0
        process = Mock()
        process.poll.return_value = None
        with patch('local_llm.subprocess.Popen', return_value=process), \
             self.transport(lambda _: self.fail('No request expected after startup timeout')), \
             self.assertLogs('local_llm', level='ERROR'):
            await self.model.start()
        process.terminate.assert_called_once()
        self.assertIsNone(self.model.process)
        self.assertIsNone(self.model.log)
        self.assertEqual(self.model.phase, 'indisponível')

    async def test_occupied_port_does_not_reuse_or_terminate_external_process(self):
        executable = Path(self.folder.name) / 'fake.exe'
        executable.touch()
        self.model.path = executable
        with socket.socket() as external:
            external.bind(('127.0.0.1', 0))
            external.listen()
            self.model.port = external.getsockname()[1]
            with patch('local_llm.subprocess.Popen') as spawn, self.assertLogs('local_llm', level='ERROR'):
                await self.model.start()
            spawn.assert_not_called()
            self.assertIn(str(self.model.port), self.model.error)
            self.assertEqual(external.getsockname()[1], self.model.port)

    async def test_shutdown_during_loading_does_not_leave_owned_process(self):
        executable = Path(self.folder.name) / 'fake.exe'
        executable.touch()
        self.model.path = executable
        self.model.port = 0
        process = Mock()
        process.poll.return_value = None
        loading = asyncio.Event()

        async def response(request):
            loading.set()
            await asyncio.Event().wait()

        with patch.object(server, 'local_llm', self.model), \
             patch('local_llm.subprocess.Popen', return_value=process), self.transport(response), \
             patch.object(server, 'stop', new=AsyncMock()):
            async with server.lifespan(server.app):
                await asyncio.wait_for(loading.wait(), timeout=2)
        process.terminate.assert_called_once()
        self.assertIsNone(self.model.process)
        self.assertIsNone(self.model.log)

    async def test_extracts_literal_claims_and_rejects_changed_numbers_and_duplicates(self):
        self.make_ready()
        transcript = 'O Brasil tem 26 estados e um Distrito Federal.'

        def response(request):
            self.assertEqual(str(request.url), self.model.url + '/v1/chat/completions')
            payload = json.loads(request.content)
            self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])
            if 'decisions' in payload['response_format']['schema']['properties']:
                return approved_review('O Brasil')
            self.assertEqual(payload['response_format']['schema']['properties']['claims']['maxItems'], 3)
            claims = [{'text': transcript}, {'text': transcript}, {'text': 'O Brasil tem 27 estados.'}, {'text': 42}]
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
                'content': json.dumps({'claims': claims})}}]})

        with self.transport(response):
            self.assertEqual(await self.model.extract_claims(transcript), [transcript])

    async def test_invalid_or_truncated_response_is_not_accepted(self):
        self.make_ready()
        for reason, content in [('length', '{"claims":[]}'), ('stop', 'not JSON'),
                                ('stop', '{"claims":"invalid"}')]:
            with self.subTest(reason=reason, content=content):
                with self.transport(lambda _: httpx.Response(200, json={'choices': [{
                    'finish_reason': reason, 'message': {'content': content}}]})):
                    with self.assertRaises(RuntimeError):
                        await self.model.extract_claims('O Brasil tem 26 estados.')

    async def test_explicit_personal_opinion_is_filtered_from_mixed_output(self):
        self.make_ready()
        fact = 'Recife é a capital de Pernambuco.'
        opinion = 'Eu prefiro morar no interior.'
        content = json.dumps({'claims': [{'text': fact}, {'text': opinion}]})
        def response(request):
            if 'decisions' in json.loads(request.content)['response_format']['schema']['properties']:
                return approved_review('Recife')
            return completion(json.loads(content))
        with self.transport(response):
            self.assertEqual(await self.model.extract_claims(fact + ' ' + opinion), [fact])

    async def test_reported_noise_is_rejected_even_if_extractor_suggests_it(self):
        self.make_ready()
        cases = json.loads((Path(__file__).parent / 'fixtures' / 'claim_selection.json').read_text(encoding='utf-8'))
        for text in cases['rejected']:
            with self.subTest(text=text), patch.object(self.model, '_complete', new=AsyncMock(
                    side_effect=[{'claims': [{'text': text}]}, {'decisions': [{
                        'id': 0, 'kind': 'uncertain', 'self_contained': False,
                        'externally_verifiable': False, 'subject': ''}]}])):
                self.assertEqual(await self.model.extract_claims(text), [])

    async def test_eligibility_requires_explicit_complete_and_verifiable_fact(self):
        self.make_ready()
        text = 'O Brasil tem 26 estados e um Distrito Federal.'
        valid = {'id': 0, 'kind': 'fact', 'self_contained': True,
                 'externally_verifiable': True, 'subject': 'O Brasil'}
        invalid = [[], [dict(valid, kind='opinion')], [dict(valid, kind='question')],
                   [dict(valid, kind='fragment')], [dict(valid, kind='uncertain')],
                   [dict(valid, self_contained=False)], [dict(valid, externally_verifiable=False)],
                   [dict(valid, externally_verifiable='true')], [dict(valid, subject='')],
                   [dict(valid, subject='outro país')], [dict(valid, id=1)], [valid, valid]]
        for decisions in invalid:
            with self.subTest(decisions=decisions), patch.object(self.model, '_complete', new=AsyncMock(
                    side_effect=[{'claims': [{'text': text}]}, {'decisions': decisions}])):
                self.assertEqual(await self.model.extract_claims(text), [])

    async def test_false_but_verifiable_claim_is_eligible_without_truth_verdict(self):
        self.make_ready()
        text = 'A Terra é plana.'
        review = {'id': 0, 'kind': 'fact', 'self_contained': True,
                  'externally_verifiable': True, 'subject': 'A Terra'}
        with patch.object(self.model, '_complete', new=AsyncMock(side_effect=[
                {'claims': [{'text': text}]}, {'decisions': [review]}])):
            self.assertEqual(await self.model.extract_claims(text), [text])

    async def test_invalid_review_discards_extraction_instead_of_bypassing_review(self):
        self.make_ready()
        text = 'Recife é a capital de Pernambuco.'
        with patch.object(self.model, '_complete', new=AsyncMock(side_effect=[
                {'claims': [{'text': text}]}, {'decisions': 'invalid'}])):
            with self.assertRaises(RuntimeError):
                await self.model.extract_claims(text)

    async def test_extractor_cannot_turn_a_question_into_a_statement(self):
        self.make_ready()
        text = 'O Brasil tem 26 estados e um Distrito Federal'
        for transcript in (text + '?', text + ' atualmente?'):
            with self.subTest(transcript=transcript), patch.object(self.model, '_complete', new=AsyncMock(
                    return_value={'claims': [{'text': text}]})) as complete:
                self.assertEqual(await self.model.extract_claims(transcript), [])
                complete.assert_awaited_once()

    async def test_dead_process_is_reported_and_no_request_is_sent(self):
        self.make_ready()
        self.model.process.poll.return_value = 1
        with self.transport(lambda _: self.fail('Dead model must not receive a request')):
            with self.assertRaises(RuntimeError):
                await self.model.extract_claims('O Brasil tem 26 estados.')
        self.assertFalse(self.model.status()['ready'])

    async def test_application_lifespan_stops_capture_and_owned_model(self):
        local = Mock(start=AsyncMock(), stop=AsyncMock())
        with patch.object(server, 'local_llm', local), patch.object(server, 'stop', new=AsyncMock()) as stop:
            async with server.lifespan(server.app):
                await asyncio.sleep(0)
                local.start.assert_awaited_once()
            stop.assert_awaited_once()
            local.stop.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
