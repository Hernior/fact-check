import asyncio
import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import server
from local_llm import LocalLlamafile


def decision(reference='Você', subject='', evidence='', **kwargs):
    return {'decisions': [{'id': 0, 'kind': 'fact', 'verifiable': True, 'self_contained': False,
                          'reference': reference, 'subject': subject, 'evidence': evidence, **kwargs}]}


class ContextualExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def extract(self, text, review, context='', participants=None):
        model = LocalLlamafile('.')
        with patch.object(model, '_complete', new=AsyncMock(side_effect=[
            {'claims': [{'text': text}]}, review])):
            return await model.extract_claims(text, debate=True, context=context,
                                              participants=participants or ['Lula', 'Flávio Bolsonaro'])

    async def test_pronoun_is_pending_and_participant_list_is_not_evidence(self):
        text = 'Você recebeu 154 milhões.'
        claims = await self.extract(text, decision(subject='Lula', evidence='Lula'))
        self.assertEqual(claims[0]['text'], text)
        self.assertEqual(claims[0]['subject'], '')

    async def test_model_name_in_reference_field_cannot_assign_a_participant(self):
        claims = await self.extract('Você recebeu 154 milhões.', decision(reference='Lula'))
        self.assertEqual(claims[0]['reference'], 'Você')
        self.assertEqual(claims[0]['subject'], '')

    async def test_explicit_context_can_identify_subject_but_ambiguous_context_cannot(self):
        evidence = 'Lula, você recebeu 154 milhões.'
        claims = await self.extract('Você recebeu 154 milhões.', decision(subject='Lula', evidence=evidence), context=evidence)
        self.assertEqual(claims[0]['text'], 'Lula recebeu 154 milhões.')
        evidence = 'Lula disse a Flávio Bolsonaro que você recebeu 154 milhões.'
        claims = await self.extract('Você recebeu 154 milhões.', decision(subject='Lula', evidence=evidence), context=evidence)
        self.assertEqual(claims[0]['subject'], '')

    async def test_question_and_opinion_remain_ineligible(self):
        for text in ['Você recebeu 154 milhões?', 'Que merda é essa?', 'Eu acho essa roupa bonita.']:
            self.assertEqual(await self.extract(text, decision()), [])

    async def test_literal_sentence_is_reviewed_when_proposer_omits_or_paraphrases_it(self):
        model = LocalLlamafile('.')
        text = 'O Brasil tem 26 estados.'
        with patch.object(model, '_complete', new=AsyncMock(side_effect=[
            {'claims': [{'text': 'O Brasil possui 26 estados.'}]}, decision(reference='', subject='O Brasil', self_contained=True)])):
            results = await model.extract_claims(text, debate=True)
        self.assertEqual(results[0]['text'], text)

    async def test_future_promise_can_be_discarded_by_semantic_review(self):
        result = await self.extract('Vocês voltarão a ser valorizados.',
                                    decision(reference='Vocês', kind='uncertain', verifiable=False))
        self.assertEqual(result, [])

    async def test_model_cannot_remove_question_mark(self):
        model = LocalLlamafile('.')
        with patch.object(model, '_complete', new=AsyncMock(return_value={'join_previous': False,
            'text': 'Você mentiu com a sociedade brasileira.'})):
            with self.assertRaises(RuntimeError):
                await model.revise_transcript('Você mentiu com a sociedade brasileira?')


class SubjectApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.database = patch.object(server, 'DB', Path(self.folder.name) / 'subjects.sqlite3')
        self.database.start()
        server.init()
        server.worker = None
        server.tab_queue = None
        server.state.update(running=False, segments=0, claims=0)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test')
        for amount in [134, 154]:
            text = f'Você recebeu {amount} milhões.'
            identifier = server.store('claims', {'text': text, 'status': 'aguardando sujeito',
                'at': server.utc(), 'offset_s': 12, 'explanation': '', 'sources': '[]', 'evidence_count': 0})
            server.store('claim_subjects', {'claim_id': identifier, 'data': json.dumps({
                'original_text': text, 'reference': 'Você', 'subject': '', 'participants': ['Lula', 'Flávio Bolsonaro'],
                'resolved_at': None, 'context': text}, ensure_ascii=False)})

    async def asyncTearDown(self):
        await server.stop()
        await self.client.aclose()
        self.database.stop()
        self.folder.cleanup()

    async def test_pending_subject_blocks_search_before_spending_quota(self):
        with patch.object(server, 'search_sources', side_effect=AssertionError('Brave called')):
            response = await self.client.post('/api/claims/1/search')
        self.assertEqual(response.status_code, 409)
        self.assertEqual(server.brave_quota()['used'], 0)

    async def test_resolution_is_per_claim_durable_and_does_not_interrupt_capture(self):
        server.state['running'] = True
        with patch.object(server, 'search_sources', side_effect=AssertionError('Brave called')):
            response = await self.client.post('/api/claims/1/subject', json={'subject': 'Flávio Bolsonaro'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(server.state['running'])
        server.init()
        feed = (await self.client.get('/api/feed')).json()
        resolved, pending = {c['id']: c for c in feed['claims']}[1], {c['id']: c for c in feed['claims']}[2]
        self.assertEqual(resolved['text'], 'Flávio Bolsonaro recebeu 134 milhões.')
        self.assertEqual(resolved['status'], 'aguardando pesquisa')
        self.assertEqual(resolved['subject_resolution']['original_text'], 'Você recebeu 134 milhões.')
        self.assertEqual(pending['text'], 'Você recebeu 154 milhões.')
        self.assertEqual(pending['status'], 'aguardando sujeito')
        self.assertEqual(server.brave_quota()['used'], 0)

    async def test_custom_subject_validation_duplicate_and_conflicting_clicks(self):
        for subject in ['', ' ', 'x', 'a' * 81]:
            self.assertEqual((await self.client.post('/api/claims/1/subject', json={'subject': subject})).status_code, 422)
        self.assertEqual((await self.client.post('/api/claims/1/subject', json={'subject': 'Outra pessoa'})).status_code, 200)
        self.assertEqual((await self.client.post('/api/claims/1/subject', json={'subject': 'Outra pessoa'})).status_code, 200)
        self.assertEqual((await self.client.post('/api/claims/1/subject', json={'subject': 'Lula'})).status_code, 409)
        self.assertEqual((await self.client.post('/api/claims/99/subject', json={'subject': 'Lula'})).status_code, 404)

    async def test_participants_are_snapshotted_before_capture_and_validated(self):
        gate = asyncio.Event()
        async def run(queue, participants):
            self.assertEqual(participants, ['Lula', 'Flávio Bolsonaro'])
            await gate.wait()
        with patch.object(server, 'run_capture', new=run):
            response = await self.client.post('/api/tab/start', json={'participants': [' Lula ', 'Lula', 'Flávio Bolsonaro']})
            self.assertEqual(response.status_code, 200)
            await asyncio.sleep(0)
            gate.set()
            await server.stop()
        for participants in [['x'], ['a' * 81], ['Nome'] * 21, [42]]:
            response = await self.client.post('/api/tab/start', json={'participants': participants})
            self.assertEqual(response.status_code, 422)

    async def test_clear_removes_subjects_but_not_quota(self):
        server.brave_quota(reserve=True)
        self.assertEqual((await self.client.post('/api/clear', json={'confirmed': True})).status_code, 200)
        with closing(server.conn()) as database:
            self.assertEqual(database.execute('SELECT COUNT(*) FROM claim_subjects').fetchone()[0], 0)
        self.assertEqual(server.brave_quota()['used'], 1)

    async def test_capture_creates_pending_subject_and_keeps_processing_next_chunk(self):
        item = {'text': 'Você recebeu 134 milhões.', 'original_text': 'Você recebeu 134 milhões.',
                'reference': 'Você', 'subject': '', 'evidence': ''}
        with patch.object(server, 'transcribe', new=AsyncMock(side_effect=['Você recebeu 134 milhões.', 'Próxima fala.'])), \
             patch.object(server.local_llm, 'revise_transcript', new=AsyncMock(side_effect=lambda text, **kwargs: {
                 'join_previous': False, 'text': text})), \
             patch.object(server.local_llm, 'extract_claims', new=AsyncMock(side_effect=[[item], []])):
            session = (await self.client.post('/api/tab/start', json={'participants': ['Lula', 'Flávio Bolsonaro']})).json()['session']
            for _ in range(2):
                await self.client.post('/api/tab/audio', params={'session': session}, content=b'\x00\x00' * 16000)
            for _ in range(100):
                if server.state['claims'] == 1:
                    break
                await asyncio.sleep(0.005)
            self.assertTrue(server.state['running'])
            self.assertEqual(server.state['segments'], 2)
            claim = (await self.client.get('/api/feed')).json()['claims'][0]
            self.assertEqual(claim['status'], 'aguardando sujeito')
            self.assertEqual(claim['subject_resolution']['participants'], ['Lula', 'Flávio Bolsonaro'])
            await server.stop()

    async def test_same_pronoun_claim_in_different_turns_has_separate_subject_choices(self):
        item = {'text': 'Você recebeu 134 milhões.', 'original_text': 'Você recebeu 134 milhões.',
                'reference': 'Você', 'subject': '', 'evidence': ''}
        with patch.object(server, 'transcribe', new=AsyncMock(return_value=item['text'])), \
             patch.object(server.local_llm, 'revise_transcript', new=AsyncMock(return_value={
                 'join_previous': False, 'text': item['text']})), \
             patch.object(server.local_llm, 'extract_claims', new=AsyncMock(return_value=[item])):
            session = (await self.client.post('/api/tab/start')).json()['session']
            for _ in range(2):
                await self.client.post('/api/tab/audio', params={'session': session}, content=b'\x00\x00' * 16000)
            while server.state['segments'] < 2:
                await asyncio.sleep(0.005)
            await server.stop()
        claims = (await self.client.get('/api/feed')).json()['claims']
        self.assertEqual(len(claims), 2)  # seeded claims belong to the archived session


if __name__ == '__main__':
    unittest.main()
