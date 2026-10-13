"""Conformance with the A2G JSON Schema in the sibling a2g-protocol repository (skipped when it is not checked out).

The service must accept every valid example, reject the invalid ones it can judge, and produce responses that
validate against the schema's $defs (checked with the protocol repo's own ajv through Node).
"""
import io
import json
import pathlib
import shutil
import subprocess
import unittest
from unittest.mock import patch

import decision_service as service
import monitor

SCHEMA = pathlib.Path(__file__).resolve().parents[2] / 'a2g-protocol' / 'schema' / '0.1'
AJV = SCHEMA.parent / 'node_modules' / 'ajv' / 'dist' / '2020.js'
NODE = shutil.which('node')


def examples(folder, definition):
    return [(path.name, json.loads(path.read_text())) for path in sorted((SCHEMA / folder / definition).glob('*.json'))]


def schema_errors(pairs):
    """Validate [(definition, instance)] with ajv; return the list of failures."""
    script = '''
import Ajv2020 from %s;
import { readFileSync } from 'node:fs';
const schema = JSON.parse(readFileSync(%s, 'utf8'));
const ajv = new Ajv2020({ allErrors: true, strict: true }); ajv.addSchema(schema);
const pairs = JSON.parse(readFileSync(0, 'utf8')); const failures = [];
for (const [definition, instance] of pairs) {
  const validate = ajv.getSchema(`${schema.$id}#/$defs/${definition}`);
  if (!validate(instance)) failures.push(definition + ': ' + ajv.errorsText(validate.errors) + ' in ' + JSON.stringify(instance).slice(0, 200));
}
console.log(JSON.stringify(failures));''' % (json.dumps(str(AJV)), json.dumps(str(SCHEMA / 'a2g.schema.json')))
    result = subprocess.run([NODE, '--input-type=module', '-e', script], input=json.dumps(pairs), capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


@unittest.skipUnless((SCHEMA / 'a2g.schema.json').is_file(), 'a2g-protocol schema not checked out next to a2g-poc')
class SchemaConformanceTests(unittest.TestCase):
    def setUp(self):
        with monitor.LOCK:
            monitor.CLIENTS.clear()
        service.CACHE.clear()

    def test_service_accepts_valid_examples_and_rejects_invalid_ones(self):
        for name, manifest in examples('examples', 'Manifest'):
            with self.subTest(manifest=name):
                service.check_manifest(manifest)
        for name, manifest in examples('invalid', 'Manifest'):
            with self.subTest(invalid_manifest=name), self.assertRaises(ValueError):
                service.check_manifest(manifest)
        for name, request in examples('examples', 'SessionRequest'):
            self.assertEqual(service.handle('/v1/session', request)[0], 200, name)
        for name, request in examples('invalid', 'SessionRequest'):
            with self.subTest(invalid_session=name), self.assertRaises(ValueError):
                service.handle('/v1/session', request)
        lango = examples('examples', 'SessionRequest')[0][1]
        manifest = monitor.manifest(lango['client_id'])
        manifests = dict(examples('examples', 'Manifest'))
        for name, request in examples('examples', 'DecideRequest'):  # each decide example goes with the manifest of the same name, else Lango's
            with self.subTest(decide=name):
                service.check_decide(request, service.check_manifest(manifests[name]) if name in manifests else manifest)
        for name, request in examples('examples', 'TelemetryRequest'):
            self.assertEqual(service.handle('/v1/telemetry', request), (200, {'ok': True}), name)
        for name, request in examples('examples', 'ContextRequest'):
            self.assertEqual(service.handle('/v1/context', request)[0], 200, name)
        service.handle('/v1/session', {'a2g': '0.1', 'client_id': 'c1', 'manifest': {'entity_types': {'g': {'actions': {'a': 'Act'}}}}})
        for name, request in examples('invalid', 'DecideRequest'):
            with self.subTest(invalid_decide=name), self.assertRaises(ValueError):
                service.check_decide(request, monitor.manifest('c1'))
        for name, request in examples('invalid', 'ContextRequest') + examples('invalid', 'ReportRequest'):
            with self.subTest(invalid=name), self.assertRaises(ValueError):
                service.handle('/v1/context' if 'scopes' in request or len(request) == 1 else '/v1/report', request | {'client_id': 'c1'})

    @unittest.skipUnless(NODE and AJV.is_file(), 'Node or the protocol repo node_modules (npm ci in a2g-protocol/schema) missing')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only'})
    @patch('decision_service.urllib.request.urlopen')
    def test_service_responses_validate_against_schema(self, urlopen):
        lango = examples('examples', 'SessionRequest')[0][1]
        pairs = [('SessionResponse', service.handle('/v1/session', lango)[1])]
        decide = examples('examples', 'DecideRequest')[0][1]
        urlopen.return_value = io.BytesIO(json.dumps({'answers': [{'name': 'entity_0', 'type': 'choice', 'choice': 'flee', 'confidence': 0.7},
                                                                  {'name': 'entity_1', 'type': 'refusal'}], 'usage': {'input_tokens': 500}}).encode())
        pairs.append(('DecideResponse', service.handle('/v1/decide', decide)[1]))
        urlopen.return_value = io.BytesIO(json.dumps({'answers': [{'name': 'entity_0', 'type': 'choice', 'choice': 'flee'}]}).encode())
        pairs.append(('DecideResponse', service.handle('/v1/decide', decide | {'entities': decide['entities'][:1]})[1]))
        urlopen.side_effect = TimeoutError()
        pairs.append(('DecideResponse', service.handle('/v1/decide', decide | {'epoch': 2})[1]))
        context = examples('examples', 'ContextRequest')[0][1]
        pairs.append(('ContextResponse', service.handle('/v1/context', context)[1]))
        pairs.append(('OkResponse', service.handle('/v1/report', dict(examples('examples', 'ReportRequest'))['executed.json'])[1]))
        pairs.append(('OkResponse', service.handle('/v1/telemetry', examples('examples', 'TelemetryRequest')[0][1])[1]))
        urlopen.side_effect = None  # decision slots with probabilities, one slot refused
        slots = {'a2g': '0.1', 'client_id': 'slots', 'manifest': next(m for name, m in examples('examples', 'Manifest') if name == 'decision-slots.json')}
        service.handle('/v1/session', slots)
        slot_request = next(r for name, r in examples('examples', 'DecideRequest') if name == 'decision-slots.json') | {'client_id': 'slots'}
        urlopen.return_value = io.BytesIO(json.dumps({'answers': [
            {'name': 'entity_0_0', 'type': 'choice', 'choice': 'push_harder', 'confidence': 0.4, 'probabilities': {'carry_on': 0.5, 'push_harder': 0.3, 'reduce_effort': 0.2}},
            {'name': 'entity_0_1', 'type': 'choice', 'choice': 'none', 'confidence': 0.9},
            {'name': 'entity_0_2', 'type': 'refusal'}]}).encode())
        slot_reply = service.handle('/v1/decide', slot_request)[1]
        pairs.append(('DecideResponse', slot_reply))
        pairs.append(('OkResponse', service.handle('/v1/report', {'client_id': 'slots', 'entity_id': 'p7', 'decision_id': slot_reply['decisions'][0]['decision_id'],
                                                                  'status': 'executed', 'actions': {'effort': 'carry_on', 'help': 'none'}})[1]))
        try:
            service.handle('/v1/decide', decide | {'client_id': 'unknown'})
        except service.Rejected as failure:
            pairs.append(('ErrorResponse', {'error': str(failure), 'code': failure.code}))
        self.assertEqual(schema_errors(pairs), [])


if __name__ == '__main__':
    unittest.main()
