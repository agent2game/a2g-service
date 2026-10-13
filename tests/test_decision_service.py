import copy
import io
import json
import time
import unittest
from unittest.mock import patch

import decision_service as service
import monitor

MANIFEST = {
    'world': 'Test arena.',
    'world_fields': {'threat': 'number'},
    'scopes': {'keep': {'text': 'A castle keep.'}, 'hall': {'parent': 'keep', 'text': 'The great hall.', 'fields': {'alarm': [1, 2, 3]}}},
    'profiles': {'captain': {'type': 'guard', 'text': 'Veteran captain.'}},
    'entity_types': {
        'guard': {
            'description': 'Careful guard.',
            'fields': {'hp_fraction': 'number', 'sees_target': 'boolean', 'mood': ['calm', 'angry'],
                       'spot': 'vector2', 'target': 'entity_ref', 'allies': {'list': {'id': 'entity_ref', 'hp': 'number'}, 'max': 2}},
            'actions': {'attack': 'Engage the target', 'hide': 'Take cover', 'wait': 'Hold position'},
            'events': {'alarm': 'An alarm was raised'},
            'decision_ttl_ms': 5000, 'refresh_ms': 8000, 'perception_radius': 50,
            'cache': {'key': ['hp_fraction', 'sees_target', 'world.threat', 'scope.alarm', 'parent.action'], 'buckets': {'hp_fraction': [0.34, 0.67]}, 'ttl_ms': 10000}},
        'squad': {'description': 'A guard squad.', 'actions': {'hold': 'Hold the line', 'push': 'Push forward'}},
        'hero': {'description': 'Player avatar', 'fields': {'hp_fraction': 'number'}, 'actions': {}, 'perception_radius': 30},
    },
}
OBSERVATION = {'hp_fraction': 0.5, 'sees_target': True, 'mood': 'calm', 'spot': [1.0, 2.0], 'target': 'p', 'allies': [{'id': 'g2', 'hp': 1.0}]}
ENTITY = {'id': 'g1', 'type': 'guard', 'observation': OBSERVATION, 'legal_actions': ['attack', 'hide']}
DECIDE = {'client_id': 'test-client', 'epoch': 3, 'world': {'threat': 2}, 'scope': 'hall', 'scope_state': {'alarm': 2}, 'entities': [ENTITY, ENTITY | {'id': 'g2'}]}


def provider_reply(*answers, usage=None):
    return io.BytesIO(json.dumps({'answers': list(answers), 'usage': usage or {'input_tokens': 1000, 'output_tokens': 0}}).encode())


def sent(urlopen, call=-1):
    payload = json.loads(urlopen.call_args_list[call][0][0].data)
    return payload, json.loads(payload['input'])


class ServiceTests(unittest.TestCase):
    def setUp(self):
        with monitor.LOCK:
            monitor.CLIENTS.clear()
            monitor.TOTAL_COST = monitor.empty_cost()
        service.CACHE.clear()
        monitor.SETTINGS.clear()
        status, reply = service.handle('/v1/session', {'a2g': '0.1', 'client_id': 'test-client', 'game': 'Test', 'engine': 'None', 'manifest': MANIFEST})
        self.assertEqual(status, 200)
        self.assertRegex(reply['manifest_hash'], r'^sha256:[0-9a-f]{64}$')

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only'})
    @patch('decision_service.urllib.request.urlopen')
    def test_decision_slots_and_probabilities(self, urlopen):
        manifest = copy.deepcopy(MANIFEST)
        manifest['entity_types']['worker'] = {'description': 'A startup employee.', 'fields': {'stress': ['low', 'high']},
            'decisions': {'effort': {'description': 'How hard they work this week', 'actions': {'carry_on': 'Work as usual', 'push_harder': 'More hours', 'reduce_effort': 'Slack off'}},
                          'concern': {'actions': {'none': 'Say nothing', 'raise_concern': 'Talk to the CEO'}},
                          'politics': {'actions': {'none': 'Nothing', 'take_credit': 'Claim a win'}}},
            'cache': {'key': ['stress']}}
        with self.assertRaises(ValueError):
            service.check_manifest(manifest | {'entity_types': {'w': manifest['entity_types']['worker'] | {'actions': {'a': 'b'}}}})
        service.handle('/v1/session', {'a2g': '0.1', 'client_id': 'slots', 'manifest': manifest})
        worker = {'id': 'w1', 'type': 'worker', 'observation': {'stress': 'high'},
                  'legal_actions': {'effort': ['carry_on', 'push_harder', 'reduce_effort'], 'concern': ['none', 'raise_concern']}}
        with self.assertRaises(ValueError):
            service.handle('/v1/decide', {'client_id': 'slots', 'epoch': 1, 'world': {'threat': 1}, 'entities': [worker | {'legal_actions': ['carry_on']}]})
        urlopen.return_value = provider_reply(
            {'name': 'entity_0_0', 'type': 'choice', 'choice': 'carry_on', 'confidence': 0.6, 'probabilities': {'carry_on': 0.6, 'push_harder': 0.3, 'reduce_effort': 0.1, 'bogus': 0.5}},
            {'name': 'entity_0_1', 'type': 'refusal'})
        decision = service.handle('/v1/decide', {'client_id': 'slots', 'epoch': 1, 'world': {'threat': 1}, 'entities': [worker]})[1]['decisions'][0]
        self.assertEqual((decision['source'], 'action' in decision), ('AI', False))
        self.assertEqual(decision['slots']['effort'], {'action': 'carry_on', 'label': 'Work as usual', 'confidence': 0.6,
                                                       'probabilities': {'carry_on': 0.6, 'push_harder': 0.3, 'reduce_effort': 0.1}})
        self.assertEqual(decision['slots']['concern']['code'], 'refused', 'One slot can fall back while the others decide')
        payload = sent(urlopen)[0]
        self.assertEqual([q['name'] for q in payload['questions']], ['entity_0_0', 'entity_0_1'], 'Only slots with legal actions are asked')
        self.assertIn('decide only "effort"', payload['questions'][0]['instructions'])
        report = {'client_id': 'slots', 'entity_id': 'w1', 'decision_id': decision['decision_id'], 'status': 'executed'}
        self.assertEqual(service.handle('/v1/report', report | {'actions': {'effort': 'dance'}}), (200, {'ok': True}))
        self.assertEqual(monitor.snapshot('slots')['stats']['reports'], 0, 'Illegal executed actions are ignored')
        service.handle('/v1/report', report | {'actions': {'effort': 'push_harder', 'concern': 'none'}})  # sampled, not the argmax
        self.assertEqual(monitor.recall('slots', 'w1')[-1]['action'], 'effort=push_harder, concern=none', 'Memory keeps what was executed')
        with self.assertRaises(ValueError):
            service.handle('/v1/report', report | {'action': 'carry_on', 'actions': {'effort': 'carry_on'}})

        urlopen.return_value = provider_reply(
            {'name': 'entity_0_0', 'type': 'choice', 'choice': 'push_harder', 'confidence': 0.9},
            {'name': 'entity_0_1', 'type': 'choice', 'choice': 'none', 'confidence': 0.8})
        first = service.handle('/v1/decide', {'client_id': 'slots', 'epoch': 2, 'world': {'threat': 1}, 'entities': [worker, worker | {'id': 'w2'}]})[1]['decisions']
        self.assertEqual([d['slots']['effort']['action'] for d in first], ['push_harder', 'push_harder'])
        self.assertTrue(first[1]['cached'], 'Equivalent workers share one answer')
        again = service.handle('/v1/decide', {'client_id': 'slots', 'epoch': 2, 'world': {'threat': 1}, 'entities': [worker | {'id': 'w3'}]})[1]['decisions'][0]
        self.assertEqual((again['cached'], again['slots']['concern']['action']), (True, 'none'))
        self.assertEqual(urlopen.call_count, 2)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only', 'A2G_QUESTIONS_PER_CALL': '16'})
    @patch('decision_service.urllib.request.urlopen')
    def test_many_slots_are_split_across_provider_calls(self, urlopen):
        slots = {'s%02d' % i: {'actions': {'none': 'Nothing', 'act': 'Do it'}} for i in range(40)}  # no cap on slots
        manifest = {'entity_types': {'worker': {'description': 'Busy.', 'decisions': slots}}}
        service.handle('/v1/session', {'a2g': '0.1', 'client_id': 'many', 'manifest': manifest})
        calls = []
        def reply(request, timeout):
            questions = json.loads(request.data)['questions']
            calls.append(len(questions))
            if len(questions) == 8:
                raise TimeoutError()  # the last, smaller call fails: only its slots fall back
            return provider_reply(*[{'name': q['name'], 'type': 'choice', 'choice': 'act', 'confidence': 0.9} for q in questions])
        urlopen.side_effect = reply
        entity = {'id': 'w1', 'type': 'worker', 'observation': {}, 'legal_actions': {slot: ['none', 'act'] for slot in slots}}
        decision = service.handle('/v1/decide', {'client_id': 'many', 'epoch': 1, 'world': {}, 'entities': [entity]})[1]['decisions'][0]
        self.assertEqual(sorted(calls), [8, 16, 16])
        decided = [slot for slot, result in decision['slots'].items() if result.get('action') == 'act']
        failed = [slot for slot, result in decision['slots'].items() if result.get('code') == 'unavailable']
        self.assertEqual((len(decided), len(failed), decision['source']), (32, 8, 'AI'))

    def test_budget_warning(self):
        tight = copy.deepcopy(MANIFEST)
        tight['entity_types']['guard']['max_input_tokens'] = 60
        self.assertEqual(service.budget_warnings(service.check_manifest(MANIFEST)), [])
        warnings = service.budget_warnings(service.check_manifest(tight))
        self.assertEqual(len(warnings), 1)
        self.assertIn('"guard"', warnings[0])
        service.handle('/v1/session', {'a2g': '0.1', 'client_id': 'tight', 'manifest': tight})
        self.assertTrue(any('max_input_tokens 60' in log['message'] for log in monitor.snapshot('tight')['logs']))

    def test_manifest_rules(self):
        manifest = service.check_manifest(MANIFEST)
        self.assertEqual(manifest['entity_types']['guard']['cache']['ttl_ms'], 10000)
        guard = MANIFEST['entity_types']['guard']
        broken = [
            {}, MANIFEST | {'entity_types': {}}, MANIFEST | {'world_fields': {'x': 'vector4'}}, MANIFEST | {'world_fields': {'a.b': 'number'}},
            MANIFEST | {'entity_types': {'g': guard | {'fields': {'l': {'list': {'inner': {'list': {'x': 'number'}, 'max': 1}}, 'max': 2}}}}},
            MANIFEST | {'entity_types': {'g': guard | {'cache': {'key': ['memory.last']}}}},
            MANIFEST | {'entity_types': {'g': guard | {'cache': {'key': ['spot']}}}},
            MANIFEST | {'entity_types': {'g': guard | {'cache': {'key': ['hp_fraction'], 'buckets': {'hp_fraction': [0.6, 0.3]}}}}},
            MANIFEST | {'entity_types': {'g': guard | {'max_input_tokens': 10}}},
            MANIFEST | {'entity_types': {'g': guard | {'unknown': 1}}},
            MANIFEST | {'entity_types': {'hero': MANIFEST['entity_types']['hero']}},
            MANIFEST | {'scopes': {'a': {'parent': 'b', 'text': ''}, 'b': {'parent': 'a', 'text': ''}}},
            MANIFEST | {'scopes': {'a': {'text': '', 'fields': {'x': 'number'}}, 'b': {'parent': 'a', 'text': '', 'fields': {'x': 'number'}}}},
            MANIFEST | {'profiles': {'p': {'type': 'dragon', 'text': ''}}},
        ]
        for index, manifest in enumerate(broken):
            with self.subTest(index=index), self.assertRaises(ValueError):
                service.check_manifest(manifest)
        same = copy.deepcopy(MANIFEST)
        same['entity_types']['guard']['decision_ttl_ms'] = 5000.0
        same['scopes'] = {}
        self.assertEqual(service.manifest_hash(same), service.manifest_hash(MANIFEST), 'Hash ignores scopes/profiles and integral floats')
        self.assertNotEqual(service.manifest_hash(MANIFEST | {'world': 'Other'}), service.manifest_hash(MANIFEST))

    def test_decide_validation(self):
        manifest = service.check_manifest(MANIFEST)
        bad = [
            DECIDE | {'scope_state': {'alarm': 4}}, DECIDE | {'scope': 'nowhere'}, {k: v for k, v in DECIDE.items() if k != 'scope'},
            DECIDE | {'entities': [ENTITY | {'type': 'hero', 'legal_actions': ['attack']}]},
            DECIDE | {'entities': [ENTITY | {'observation': OBSERVATION | {'spot': [1, 2, 3]}}]},
            DECIDE | {'entities': [ENTITY | {'observation': OBSERVATION | {'allies': [{'id': 'x', 'hp': 1}] * 3}}]},
            DECIDE | {'entities': [ENTITY | {'observation': OBSERVATION | {'mood': 'sad'}}]},
            DECIDE | {'entities': [ENTITY | {'trigger': 'earthquake'}]},
            DECIDE | {'entities': [ENTITY | {'profile': 'nobody'}]},
            DECIDE | {'entities': [{'id': 's1', 'type': 'squad', 'observation': {}, 'legal_actions': ['hold'], 'profile': 'captain'}]},
            DECIDE | {'entities': [ENTITY, ENTITY]},
        ]
        for index, request in enumerate(bad):
            with self.subTest(index=index), self.assertRaises(ValueError):
                service.check_decide(request, manifest)
        self.assertEqual(service.check_decide(DECIDE | {'epoch': 3.0}, manifest)[0], 3)
        with self.assertRaises(service.Rejected) as caught:
            service.handle('/v1/decide', DECIDE | {'client_id': 'no-session'})
        self.assertEqual(caught.exception.status, 409)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only'})
    @patch('decision_service.urllib.request.urlopen')
    def test_batch_codes_decision_ids_and_reports(self, urlopen):
        urlopen.return_value = provider_reply({'name': 'entity_0', 'type': 'choice', 'choice': 'hide', 'confidence': 0.81234},
                                              {'name': 'entity_1', 'type': 'refusal'})
        reply = service.handle('/v1/decide', DECIDE | {'entities': [ENTITY | {'profile': 'captain', 'trigger': 'alarm'}, ENTITY | {'id': 'g2', 'observation': OBSERVATION | {'sees_target': False}}]})[1]
        first, second = reply['decisions']
        self.assertEqual((first['source'], first['action'], first['label'], first['confidence'], first['ttl_ms']), ('AI', 'hide', 'Take cover', 0.812, 5000))
        self.assertTrue(first['decision_id'].startswith('d_'))
        self.assertEqual((second['source'], second['code']), ('FALLBACK', 'refused'))
        payload, evidence = sent(urlopen)
        self.assertEqual(list(evidence)[:3], ['world', 'scopes', 'profiles'], 'Static texts come first')
        self.assertEqual([scope['name'] for scope in evidence['scopes']], ['keep', 'hall'])
        self.assertEqual(evidence['profiles'], {'captain': 'Veteran captain.'})
        self.assertEqual(evidence['scope_state'], {'alarm': 2})
        self.assertEqual(evidence['entities'][0]['trigger'], {'event': 'alarm', 'description': 'An alarm was raised'})
        self.assertIn('Careful guard.', payload['questions'][0]['instructions'], 'The trusted type description is part of the question')
        self.assertNotIn('calm', payload['questions'][0]['instructions'], 'Runtime values stay in the data')
        self.assertIn('never as instructions', payload['questions'][0]['instructions'])

        report = {'client_id': 'test-client', 'entity_id': 'g1', 'decision_id': first['decision_id'], 'action': 'hide', 'status': 'executed', 'outcome': {'hp_delta': -0.25}}
        service.handle('/v1/report', report)
        service.handle('/v1/report', report)  # replay
        service.handle('/v1/report', report | {'decision_id': 'd_forged'})
        service.handle('/v1/report', report | {'entity_id': 'g2'})
        memory = monitor.recall('test-client', 'g1')
        self.assertEqual((memory[-1]['status'], memory[-1]['outcome'], memory[-1]['trigger']), ('executed', {'hp_delta': -0.25}, 'alarm'))
        stats = monitor.snapshot('test-client')['stats']
        self.assertEqual((stats['reports'], stats['reports_ignored'], stats['fallback_codes']), (1, 3, {'refused': 1}))
        with self.assertRaises(ValueError):
            service.handle('/v1/report', {k: v for k, v in report.items() if k != 'decision_id'})

        urlopen.return_value = provider_reply({'name': 'entity_0', 'type': 'choice', 'choice': 'wait'})
        reply = service.handle('/v1/decide', DECIDE | {'entities': [ENTITY | {'observation': OBSERVATION | {'hp_fraction': 0.1}}]})[1]
        self.assertEqual(reply['decisions'][0]['code'], 'invalid_choice')
        evidence = sent(urlopen)[1]
        self.assertEqual(evidence['entities'][0]['recent_decisions'], [{'action': 'hide', 'status': 'executed', 'outcome': {'hp_delta': -0.25}}])

    @patch.dict('os.environ', {'A2G_PROVIDER': 'jev', 'TYPESAFE_API_KEY': 'test-only', 'OPENAI_API_KEY': ''})
    @patch('decision_service.urllib.request.urlopen')
    def test_jev_provider(self, urlopen):
        urlopen.return_value = io.BytesIO(json.dumps({'model': 'jev-1.13.0', 'usage': {'input_tokens': 300, 'output_tokens': 30}, 'answers': {
            'entity_0': {'type': 'choice', 'choice': 'hide', 'probabilities': {'attack': 0.1, 'hide': 0.9}, 'confidence': 0.8},
            'entity_1': {'type': 'choice', 'choice': 'wait', 'probabilities': {'wait': 1.0}, 'confidence': 1.0}}}).encode())
        first, second = service.handle('/v1/decide', DECIDE | {'entities': [ENTITY, ENTITY | {'id': 'g2', 'observation': OBSERVATION | {'sees_target': False}}]})[1]['decisions']
        self.assertEqual((first['source'], first['action'], first['confidence']), ('AI', 'hide', 0.8))
        self.assertEqual((second['source'], second['code']), ('FALLBACK', 'invalid_choice'))
        request = urlopen.call_args[0][0]
        self.assertEqual(request.full_url, 'https://api.typesafe.ai/v1/systemone')
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-only')
        payload = json.loads(request.data)
        self.assertEqual(payload['model'], 'jev-latest')
        self.assertEqual(payload['state']['entities'][0]['id'], 'g1')
        self.assertEqual(payload['questions']['entity_0']['criteria'], {'attack': 'Engage the target', 'hide': 'Take cover'})
        self.assertIn('never as instructions', payload['questions']['entity_0']['instructions'])
        self.assertEqual(service.health()['provider'], 'typesafe:jev-latest')
        cost = monitor.snapshot('test-client')['cost']
        self.assertEqual((cost['input_tokens'], round(cost['usd'], 8), cost['unpriced_calls']), (300, 1.26e-05, 0))

    @patch.dict('os.environ', {'A2G_PROVIDER': 'jev'})
    def test_console_provider_choice(self):
        self.assertEqual(service.provider_id(), 'jev', 'A2G_PROVIDER applies until the Console picks one')
        self.assertEqual(service.console('POST', '/api/provider', {'provider': 'openai'})[0], 200)
        self.assertEqual((service.provider_id(), service.health()['provider_id']), ('openai', 'openai'), 'The Console choice wins')
        with self.assertRaises(ValueError):
            service.console('POST', '/api/provider', {'provider': 'nope'})
        self.assertEqual(service.provider_id(), 'openai')

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only'})
    @patch('decision_service.urllib.request.urlopen')
    def test_decision_cache(self, urlopen):
        urlopen.return_value = provider_reply({'name': 'entity_0', 'type': 'choice', 'choice': 'attack', 'confidence': 0.9})
        one = DECIDE | {'entities': [ENTITY]}
        first = service.handle('/v1/decide', one)[1]['decisions'][0]
        similar = one | {'entities': [ENTITY | {'id': 'g9', 'observation': OBSERVATION | {'hp_fraction': 0.6, 'mood': 'angry', 'spot': [9, 9]}}]}
        second = service.handle('/v1/decide', similar)[1]['decisions'][0]
        self.assertEqual(urlopen.call_count, 1, 'Same bucket and key values reuse the decision')
        urlopen.return_value = provider_reply({'name': 'entity_0', 'type': 'choice', 'choice': 'hide'})
        twins = one | {'world': {'threat': 9}, 'entities': [ENTITY | {'id': 't1'}, ENTITY | {'id': 't2', 'observation': OBSERVATION | {'hp_fraction': 0.4}}]}
        pair = service.handle('/v1/decide', twins)[1]['decisions']
        self.assertEqual(len(sent(urlopen)[0]['questions']), 1, 'Equivalent entities in one batch are asked once')
        self.assertEqual([(d['action'], d.get('cached', False)) for d in pair], [('hide', False), ('hide', True)])
        self.assertNotEqual(pair[0]['decision_id'], pair[1]['decision_id'])
        self.assertEqual((second['action'], second['cached'], second['source']), ('attack', True, 'AI'))
        self.assertNotEqual(second['decision_id'], first['decision_id'])
        for change in [one | {'entities': [ENTITY | {'observation': OBSERVATION | {'hp_fraction': 0.9}}]}, one | {'world': {'threat': 3}},
                       one | {'scope_state': {'alarm': 3}}, one | {'entities': [ENTITY | {'trigger': 'alarm'}]},
                       one | {'entities': [ENTITY | {'legal_actions': ['attack', 'hide', 'wait']}]}, one | {'entities': [ENTITY | {'profile': 'captain'}]}]:
            urlopen.return_value = provider_reply({'name': 'entity_0', 'type': 'choice', 'choice': 'attack'})
            service.handle('/v1/decide', change)
        self.assertEqual(urlopen.call_count, 8, 'Key values, bucket, scope state, trigger, legal set and profile all split the cache')
        service.handle('/v1/context', {'client_id': 'test-client', 'scopes': {'hall': {'parent': 'keep', 'text': 'The hall is on fire.', 'fields': {'alarm': [1, 2, 3]}}}})
        urlopen.return_value = provider_reply({'name': 'entity_0', 'type': 'choice', 'choice': 'hide'})
        self.assertNotIn('cached', service.handle('/v1/decide', one)[1]['decisions'][0], 'Edited scope text never reuses old decisions')
        stats = monitor.snapshot('test-client')['stats']
        self.assertEqual((stats['cached'], stats['provider_calls']), (2, 9))
        unsure = one | {'world': {'threat': 42}}
        urlopen.return_value = provider_reply({'name': 'entity_0', 'type': 'choice', 'choice': 'hide', 'confidence': 0.12})
        service.handle('/v1/decide', unsure)
        urlopen.return_value = provider_reply({'name': 'entity_0', 'type': 'choice', 'choice': 'attack', 'confidence': 0.9})
        self.assertNotIn('cached', service.handle('/v1/decide', unsure)[1]['decisions'][0], 'Low-confidence decisions are not cached')
        urlopen.reset_mock()
        with patch('decision_service.time.time', return_value=time.time() + 11):
            service.handle('/v1/decide', one)
        self.assertEqual(urlopen.call_count, 1, 'Expired cache entries are not reused')

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only'})
    @patch('decision_service.urllib.request.urlopen')
    def test_budget_and_failures(self, urlopen):
        guard = MANIFEST['entity_types']['guard'] | {'max_input_tokens': 50}
        service.handle('/v1/session', {'a2g': '0.1', 'client_id': 'tight', 'manifest': MANIFEST | {'entity_types': {'guard': guard}}})
        reply = service.handle('/v1/decide', DECIDE | {'client_id': 'tight', 'entities': [ENTITY]})[1]
        self.assertEqual(reply['decisions'][0]['code'], 'budget')
        self.assertEqual(urlopen.call_count, 0, 'Over-budget entities are not sent')
        urlopen.side_effect = TimeoutError('timed out')
        reply = service.handle('/v1/decide', DECIDE)[1]
        self.assertEqual({d['code'] for d in reply['decisions']}, {'unavailable'})
        self.assertEqual(monitor.snapshot('test-client')['cost']['calls'], 0, 'Failed calls report no usage')
        with patch.object(service.SLOTS, 'acquire', return_value=False):
            self.assertEqual({d['code'] for d in service.handle('/v1/decide', DECIDE)[1]['decisions']}, {'busy'})

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only'})
    @patch('decision_service.urllib.request.urlopen')
    def test_context_updates(self, urlopen):
        ok = service.handle('/v1/context', {'client_id': 'test-client', 'scopes': {'yard': {'parent': 'keep', 'text': 'Courtyard.', 'fields': {'rain': 'boolean'}}},
                                            'profiles': {'captain': None, 'rookie': {'type': 'guard', 'text': 'New recruit.'}}})[1]
        self.assertEqual(ok, {'ok': True, 'scopes': 3, 'profiles': 1})
        for bad in [{'client_id': 'test-client'}, {'client_id': 'test-client', 'scopes': {'keep': None}},
                    {'client_id': 'test-client', 'scopes': {'x': {'parent': 'missing', 'text': ''}}},
                    {'client_id': 'test-client', 'profiles': {'p': {'text': 'x' * 1001}}},
                    {'client_id': 'test-client', 'profiles': {'p': {'type': 'dragon', 'text': ''}}}]:
            with self.assertRaises(ValueError):
                service.handle('/v1/context', bad)
        self.assertEqual(len(monitor.manifest('test-client')['scopes']), 3, 'A rejected update changes nothing')
        urlopen.return_value = provider_reply({'name': 'entity_0', 'type': 'choice', 'choice': 'hide'})
        service.handle('/v1/decide', DECIDE | {'scope': 'yard', 'scope_state': {'rain': True}, 'entities': [ENTITY | {'profile': 'rookie'}]})
        self.assertEqual(sent(urlopen)[1]['profiles'], {'rookie': 'New recruit.'})
        with self.assertRaises(ValueError):
            service.handle('/v1/decide', DECIDE | {'entities': [ENTITY | {'profile': 'captain'}]})

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only'})
    @patch('decision_service.urllib.request.urlopen')
    def test_situation_groups_and_session_scope(self, urlopen):
        guard = {'type': 'guard', 'state': 'IDLE', 'fields': {'hp_fraction': 1.0}, 'decision': {}}
        entities = [guard | {'id': 'g1', 'position': [0, 0], 'parent': 's1'}, guard | {'id': 'g2', 'position': [20, 0], 'parent': 's1'},
                    guard | {'id': 'g3', 'position': [200, 0]}, {'id': 'p', 'type': 'hero', 'fields': {'hp_fraction': 0.5}, 'position': [10.0, 0.0]}]
        telemetry = {'client_id': 'test-client', 'mode': 'ai', 'scene': 'L1', 'paused': False, 'world': {}}
        monitor.telemetry(telemetry | {'entities': entities})
        monitor.telemetry(telemetry | {'entities': entities[:2] + entities[3:]})
        service.handle('/v1/session', {'a2g': '0.1', 'client_id': 'other', 'manifest': MANIFEST})
        monitor.telemetry(telemetry | {'client_id': 'other', 'entities': [guard | {'id': 'spy', 'position': [1, 0]}]})
        squad = {'id': 's1', 'type': 'squad', 'observation': {}, 'legal_actions': ['hold', 'push']}
        urlopen.return_value = provider_reply({'name': 'entity_0', 'type': 'choice', 'choice': 'push'})
        service.handle('/v1/decide', DECIDE | {'scope': None, 'scope_state': None, 'entities': [squad]} | {'scope': 'keep'})
        members = sent(urlopen)[1]['entities'][0]['members']
        self.assertEqual([m['id'] for m in members], ['g1', 'g2'])
        urlopen.return_value = provider_reply({'name': 'entity_0', 'type': 'choice', 'choice': 'hide'})
        service.handle('/v1/decide', DECIDE | {'entities': [ENTITY | {'parent': 's1'}]})
        evidence = sent(urlopen)[1]
        self.assertEqual(evidence['population'], {'guard': {'alive': 2, 'seen': 3, 'gone': 1}, 'hero': {'alive': 1, 'seen': 1, 'gone': 0}})
        entity = evidence['entities'][0]
        self.assertEqual([n['id'] for n in entity['nearby']], ['p', 'g2'], 'Nearby is session-scoped: no "spy" from another client')
        self.assertEqual((entity['parent']['id'], entity['parent']['action'], entity['parent']['label']), ('s1', 'push', 'Push forward'))
        self.assertEqual(evidence['observed_entities'][0]['nearby_counts'], {'guard': 2})
        self.assertIn('input_tokens', monitor.snapshot('test-client')['context']['g1'])

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only'})
    @patch('decision_service.urllib.request.urlopen')
    def test_database_override_and_delete(self, urlopen):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            try:
                monitor.open_db(folder + '/a2g.db')
                service.handle('/v1/session', {'a2g': '0.1', 'client_id': 'test-client', 'game': 'Test', 'engine': 'None', 'manifest': MANIFEST})
                urlopen.return_value = provider_reply({'name': 'entity_0', 'type': 'choice', 'choice': 'hide', 'confidence': 0.9},
                                                      {'name': 'entity_1', 'type': 'choice', 'choice': 'hide', 'confidence': 0.9})
                service.handle('/v1/decide', copy.deepcopy(DECIDE))
                edited = copy.deepcopy(MANIFEST)
                edited['entity_types']['guard']['description'] = 'Reckless guard.'
                with self.assertRaises(ValueError):
                    service.console('POST', '/api/override', {'game': 'Test', 'manifest': {'entity_types': {}}})
                self.assertEqual(service.console('POST', '/api/override', {'game': 'Test', 'manifest': edited})[0], 200)
                self.assertEqual(monitor.manifest('test-client')['entity_types']['guard']['description'], 'Reckless guard.')
                self.assertEqual(monitor.snapshot('test-client')['manifest_raw'], edited)
                service.console('POST', '/api/provider', {'provider': 'jev'})
                with monitor.LOCK:  # restart: summary, all-time cost, override and provider come back from the file
                    monitor.CLIENTS.clear()
                    monitor.OVERRIDES.clear()
                    monitor.SETTINGS.clear()
                monitor.open_db(folder + '/a2g.db')
                restored = monitor.snapshot('test-client')
                self.assertEqual((restored['stats']['ai'], restored['cost']['calls'], monitor.TOTAL_COST['calls']), (2, 1, 1))
                self.assertIsNotNone(restored['override'])
                self.assertEqual(service.provider_id(), 'jev')
                service.handle('/v1/session', {'a2g': '0.1', 'client_id': 'new-client', 'game': 'Test', 'engine': 'None', 'manifest': MANIFEST})
                self.assertEqual(monitor.manifest('new-client')['entity_types']['guard']['description'], 'Reckless guard.', 'Override applies to new sessions of the game')
                service.console('POST', '/api/override', {'game': 'Test', 'manifest': None})
                self.assertEqual(monitor.manifest('new-client')['entity_types']['guard']['description'], 'Careful guard.')
                self.assertEqual(service.console('DELETE', '/api/clients/test-client', {}), (200, {'deleted': 1}))
                self.assertIsNone(monitor.snapshot('test-client'))
                self.assertEqual(monitor.DB.execute('SELECT COUNT(*) FROM clients').fetchone()[0], 1)
                self.assertEqual(monitor.TOTAL_COST['calls'], 1, 'Deleting a session keeps its spend in the all-time total')
            finally:
                monitor.DB.close()
                monitor.DB = None
                monitor.OVERRIDES.clear()

    def test_local_profile_request_checks(self):
        good = {'Host': '127.0.0.1:8787', 'Content-Type': 'application/json'}
        self.assertIsNone(service.local_request_problem(good, 'POST'))
        self.assertIsNone(service.local_request_problem(good | {'Host': 'localhost:8787', 'Origin': 'http://localhost:8787'}, 'POST'))
        self.assertIsNone(service.local_request_problem({'Host': '[::1]:8787'}, 'GET'))
        for headers, method in [(good | {'Host': 'evil.example:8787'}, 'POST'), (good | {'Origin': 'https://evil.example'}, 'POST'),
                                (good | {'Content-Type': 'text/plain'}, 'POST'), ({'Host': 'rebind.example'}, 'GET'), ({}, 'GET')]:
            self.assertIsNotNone(service.local_request_problem(headers, method), headers)

    def test_http_layer(self):
        from fastapi.testclient import TestClient
        http = TestClient(service.app, base_url='http://127.0.0.1:8787')
        health = http.get('/health')
        self.assertEqual((health.status_code, health.json()['a2g'], health.headers['X-Content-Type-Options']), (200, '0.1', 'nosniff'))
        self.assertIn('text/html', http.get('/').headers['Content-Type'])
        self.assertEqual(http.get('/api/clients/test-client').status_code, 200)
        self.assertEqual(http.get('/api/clients/nobody').json(), {'error': 'Client not found'})
        self.assertEqual(http.get('/nowhere').status_code, 404)
        session = {'a2g': '0.1', 'client_id': 'http-client', 'manifest': MANIFEST}
        reply = http.post('/v1/session', json=session)
        self.assertEqual((reply.status_code, reply.json()['ok']), (200, True))
        self.assertEqual(http.post('/v1/decide', json=DECIDE | {'client_id': 'nobody'}).json()['code'], 'unknown_session')
        self.assertEqual(http.post('/v1/unknown', json={}).status_code, 404)
        self.assertEqual(http.post('/v1/session', content=b'[1]', headers={'Content-Type': 'application/json'}).json()['error'], 'Expected a JSON object')
        self.assertEqual(http.post('/v1/session', content=b'{', headers={'Content-Type': 'application/json'}).status_code, 400)
        self.assertEqual(http.post('/v1/session', content=b'x' * (service.MAX_BODY + 1), headers={'Content-Type': 'application/json'}).json()['error'], 'Invalid request size')
        self.assertEqual(http.post('/v1/session', content=json.dumps(session), headers={'Content-Type': 'text/plain'}).status_code, 403)
        self.assertEqual(http.get('/health', headers={'Host': 'rebind.example'}).status_code, 403)
        self.assertEqual(http.request('DELETE', '/api/clients/http-client', json={}).json(), {'deleted': 1})
        game = {'Origin': 'http://localhost:5173'}
        self.assertEqual(http.post('/v1/session', json=session, headers=game).status_code, 403, 'Unlisted origins stay blocked')
        with patch.dict('os.environ', {'A2G_ALLOWED_ORIGINS': 'http://localhost:5173, http://127.0.0.1:3000'}):
            preflight = http.options('/v1/decide', headers=game | {'Access-Control-Request-Method': 'POST'})
            self.assertEqual((preflight.status_code, preflight.headers['Access-Control-Allow-Origin'], preflight.headers['Vary']), (204, 'http://localhost:5173', 'Origin'))
            reply = http.post('/v1/session', json=session, headers=game)
            self.assertEqual((reply.status_code, reply.headers['Access-Control-Allow-Origin']), (200, 'http://localhost:5173'))
            self.assertEqual(http.get('/health', headers=game).headers['Access-Control-Allow-Origin'], 'http://localhost:5173')
            self.assertEqual(http.post('/api/provider', json={'provider': 'jev'}, headers=game).status_code, 403, 'Console routes keep their own origin only')
            self.assertNotIn('Access-Control-Allow-Origin', http.options('/v1/decide', headers={'Origin': 'http://localhost:9999'}).headers)
            self.assertEqual(http.options('/v1/decide', headers=game | {'Host': 'rebind.example'}).status_code, 403)
        for bad in ('*', 'http://evil.example:5173', 'http://localhost:5173/app', 'localhost:5173'):
            with patch.dict('os.environ', {'A2G_ALLOWED_ORIGINS': bad}), self.assertRaises(ValueError):
                service.allowed_origins()
        with patch.dict('os.environ', {'A2G_CONSOLE': 'off'}):  # headless: the protocol keeps working, the Console is gone
            self.assertEqual([http.get(path).status_code for path in ('/', '/api/clients')], [404, 404])
            self.assertEqual(http.post('/api/provider', json={'provider': 'jev'}).status_code, 404)
            self.assertEqual((http.get('/health').json()['console'], http.post('/v1/session', json=session).status_code), (False, 200))


if __name__ == '__main__':
    unittest.main()
