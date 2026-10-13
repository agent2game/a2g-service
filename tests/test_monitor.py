import math
import unittest
from unittest.mock import patch

import monitor

ENTITY = {'id': 'e1', 'type': 'guard', 'name': 'Guard 1', 'state': 'IDLE', 'pending': False,
          'fields': {'hp_fraction': 1.0, 'alert': False, 'zone': 'gate'}, 'decision': {}}
TELEMETRY = {'client_id': 'test-client', 'mode': 'native', 'scene': 'Level1', 'paused': False,
             'world': {'threat': 1}, 'entities': [ENTITY], 'last_error': ''}


class MonitorTests(unittest.TestCase):
    def setUp(self):
        with monitor.LOCK:
            monitor.CLIENTS.clear()

    def test_generic_telemetry_and_expiry(self):
        with patch('monitor.time.time', return_value=100):
            monitor.telemetry(TELEMETRY | {'entities': [ENTITY | {'fields': ENTITY['fields'] | {'vector': [1, 2], 'nan': math.nan, 'bad key!': 1}}]})
            client = monitor.snapshot('test-client')
            self.assertEqual(client['status'], 'online')
            self.assertEqual(client['telemetry']['entities'][0]['fields'], ENTITY['fields'], 'Only valid scalar fields are kept')
            monitor.telemetry(TELEMETRY | {'mode': 'ai', 'scene': 'Level2', 'entities': [ENTITY | {'state': 'CHASE'}]})
            logs = monitor.snapshot('test-client')['logs']
            self.assertTrue(any(log['message'] == 'scene: Level2' for log in logs))
            self.assertTrue(any(log['kind'] == 'execution' and log['entity'] == 'e1' and 'CHASE' in log['message'] for log in logs))
        with patch('monitor.time.time', return_value=108):
            self.assertEqual(monitor.snapshot('test-client')['status'], 'stale')
        with patch('monitor.time.time', return_value=120):
            self.assertEqual(monitor.snapshot('test-client')['status'], 'offline')

    def test_memory_is_bounded_and_cleared(self):
        monitor.telemetry(TELEMETRY)
        for index in range(12):
            monitor.request_started('test-client', [{'id': 'e1'}])
            monitor.request_finished('test-client', [{'id': 'e1', 'source': 'AI', 'action': 'wait', 'label': 'Hold', 'confidence': 0.5, 'decision_id': 'd_%d' % index, 'legal': ['wait', 'flee']}], 100)
        self.assertEqual(len(monitor.recall('test-client', 'e1')), monitor.MEMORY_SIZE)
        self.assertFalse(monitor.report('test-client', 'e1', 'd_5', 'dance', 'executed', {}), 'Actions that were not legal are ignored')
        self.assertTrue(monitor.report('test-client', 'e1', 'd_5', 'wait', 'executed', {'hp_delta': -1}))
        self.assertFalse(monitor.report('test-client', 'e1', 'd_5', 'wait', 'executed', {}), 'Replayed reports are ignored')
        self.assertFalse(monitor.report('test-client', 'e2', 'd_6', 'wait', 'executed', {}), 'Reports for another entity are ignored')
        self.assertEqual([item['status'] for item in monitor.recall('test-client', 'e1') if item['decision_id'] == 'd_5'], ['executed'])
        monitor.telemetry(TELEMETRY | {'entities': []})
        self.assertEqual(monitor.recall('test-client', 'e1'), [], 'Memory is dropped when the entity leaves')

    def test_bounded_storage_and_secrets(self):
        monitor.telemetry(TELEMETRY | {'OPENAI_API_KEY': 'must-not-store'})
        self.assertNotIn('must-not-store', str(monitor.snapshot('test-client')))
        for index in range(200):
            monitor.request_started('test-client', [{'id': 'e1'}])
            monitor.request_finished('test-client', [{'id': 'e1', 'source': 'FALLBACK', 'code': 'unavailable', 'reason': 'x'}], 50)
        client = monitor.snapshot('test-client')
        self.assertEqual((client['stats']['batches'], client['stats']['pending'], client['stats']['fallback']), (200, 0, 200))
        self.assertEqual(len(client['logs']), monitor.MAX_LOGS)
        for data in [TELEMETRY | {'client_id': '../bad'}, TELEMETRY | {'paused': 'no'}, TELEMETRY | {'entities': [ENTITY] * 129}]:
            with self.assertRaises(ValueError):
                monitor.telemetry(data)
        for index in range(100):  # one client per match on a game server: no cap, nobody active is dropped
            monitor.telemetry(TELEMETRY | {'client_id': 'client-%d' % index})
        self.assertEqual(len([c for c in monitor.snapshot()['clients'] if c['id'].startswith('client-')]), 100)
        with monitor.LOCK:
            monitor.CLIENTS['client-0']['last_seen'] -= monitor.IDLE_DROP_S + 1
        monitor.telemetry(TELEMETRY | {'client_id': 'client-new'})
        self.assertIsNone(monitor.snapshot('client-0'), 'Clients idle for an hour leave memory')


if __name__ == '__main__':
    unittest.main()
