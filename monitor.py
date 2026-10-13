"""Bounded, in-memory sessions, telemetry, entity memory, groups, costs and logs for A2G game clients.

Everything a client sees (memory, nearby, members, population) is derived only from that client's own
requests and telemetry: session-scoped context, as the A2G security spec requires.
"""
import copy
import json
import math
import re
import sqlite3
import threading
import time
from collections import deque

LOCK = threading.RLock()
# Live state (telemetry, memory, logs, context) lives in memory. Session summaries, every priced provider call and
# manifest overrides go to a local SQLite file when open_db() was called (the service does; tests stay in memory).
DB = None
SETTINGS = {}  # console settings (e.g. the active provider), persisted in SQLite
OVERRIDES = {}  # game name -> {'raw', 'manifest', 'hash', 'at'}: console-edited manifest used instead of the game's
# No cap on concurrent clients: a server runs one client per match. Clients idle this long leave memory (their summary
# and costs stay in SQLite). ponytail: idle sweep on new clients; per-client memory is bounded (logs, entities, memory).
IDLE_DROP_S = 3600
RESTORE_CLIENTS = 200  # session summaries reloaded at startup for the console
MAX_LOGS = 300
MAX_ENTITIES = 128
MAX_FIELDS = 24
MEMORY_SIZE = 8
MAX_ISSUED = 4096
CLIENTS = {}
SEQUENCE = 0
STARTED = time.time()
CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def text(value, limit=160):
    """Game-provided text: bounded, control characters removed (prompt-injection hygiene)."""
    if not isinstance(value, str):
        raise ValueError('Expected text')
    return CONTROL.sub('', value)[:limit]


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e9:
        raise ValueError('Invalid number')
    return value


def key(value, limit=80):
    """Identifier (client and entity ids, limit 80) or Name (types, actions, events, scopes, profiles, limit 40)."""
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,%d}' % limit, value):
        raise ValueError('Invalid identifier: ' + str(value)[:40])
    return value


def field_name(value):
    """Field names have no dots, so cache keys like world.x stay unambiguous."""
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', value):
        raise ValueError('Invalid field name: ' + str(value)[:40])
    return value


def client_id(value):
    return key(value)


def scalar(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return text(value, 80)
    return number(value)


def scalars(data, limit=MAX_FIELDS, names=field_name):
    """Keep at most `limit` scalar fields with valid names; drop everything else."""
    if not isinstance(data, dict):
        return {}
    clean = {}
    for name, value in data.items():
        if len(clean) >= limit:
            break
        try:
            clean[names(name)] = scalar(value)
        except ValueError:
            continue
    return clean


def position(value):
    """Optional entity position: a list of 2 or 3 numbers (any world units)."""
    if value is None:
        return None
    if not isinstance(value, list) or len(value) not in (2, 3):
        raise ValueError('position must be a list of 2 or 3 numbers')
    return [number(axis) for axis in value]


def empty_cost():
    return {'usd': 0.0, 'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'unpriced_calls': 0, 'by_provider': {}}


def add_cost(total, cost):
    """Accumulate one provider call: {provider, input_tokens, output_tokens, usd, priced}."""
    for bucket in (total, total['by_provider'].setdefault(cost['provider'], {'usd': 0.0, 'calls': 0, 'input_tokens': 0, 'output_tokens': 0})):
        bucket['usd'] += cost['usd']
        bucket['calls'] += 1
        bucket['input_tokens'] += cost['input_tokens']
        bucket['output_tokens'] += cost['output_tokens']
    if not cost['priced']:
        total['unpriced_calls'] += 1


TOTAL_COST = empty_cost()  # all clients, all time when the database is open; survives client eviction and deletion


def open_db(path):
    """Open (or create) the SQLite store and restore cost totals, overrides and the most recent session summaries."""
    global DB, TOTAL_COST
    db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)  # autocommit; all access under LOCK
    db.executescript('''PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS clients(id TEXT PRIMARY KEY, game TEXT, engine TEXT, protocol TEXT, manifest TEXT,
            manifest_raw TEXT, manifest_hash TEXT, first_seen REAL, last_seen REAL, stats TEXT, cost TEXT);
        CREATE TABLE IF NOT EXISTS costs(at REAL, client_id TEXT, game TEXT, provider TEXT, input_tokens INTEGER,
            output_tokens INTEGER, usd REAL, priced INTEGER);
        CREATE TABLE IF NOT EXISTS overrides(game TEXT PRIMARY KEY, raw TEXT, manifest TEXT, hash TEXT, at REAL);
        CREATE TABLE IF NOT EXISTS settings(name TEXT PRIMARY KEY, value TEXT);''')
    with LOCK:
        DB, TOTAL_COST = db, empty_cost()
        for provider, i, o, usd, priced in db.execute('SELECT provider, input_tokens, output_tokens, usd, priced FROM costs'):
            add_cost(TOTAL_COST, {'provider': provider, 'input_tokens': i, 'output_tokens': o, 'usd': usd, 'priced': bool(priced)})
        SETTINGS.update(db.execute('SELECT name, value FROM settings').fetchall())
        for game, raw, manifest, digest, at in db.execute('SELECT game, raw, manifest, hash, at FROM overrides'):
            OVERRIDES[game] = {'raw': json.loads(raw), 'manifest': json.loads(manifest), 'hash': digest, 'at': at}
        rows = db.execute('SELECT id, game, engine, protocol, manifest, manifest_raw, manifest_hash, first_seen, last_seen, stats, cost '
                          'FROM clients ORDER BY last_seen DESC LIMIT ?', (RESTORE_CLIENTS,)).fetchall()
        for identity, game, engine, protocol, manifest, raw, digest, first, last, stats, cost in rows:
            client = get_client(identity, restored=True)
            client.update({'game': game, 'engine': engine, 'protocol': protocol, 'manifest': json.loads(manifest or 'null'),
                           'manifest_hash': digest, 'first_seen': first, 'last_seen': last,
                           'stats': dict(empty_stats(), **json.loads(stats)) | {'pending': 0}, 'cost': json.loads(cost)})
            client['game_manifest'] = {'raw': json.loads(raw or 'null'), 'manifest': client['manifest'], 'hash': digest}
            event(client, 'state', 'Restored from the database (summary and cost only; telemetry, memory and logs start empty)')


def persist(client):
    """Upsert the session summary. Caller holds LOCK."""
    client['saved_at'] = time.time()
    if DB is None:
        return
    original = client.get('game_manifest') or {}
    DB.execute('INSERT OR REPLACE INTO clients VALUES (?,?,?,?,?,?,?,?,?,?,?)',
               (client['id'], client['game'], client['engine'], client['protocol'], json.dumps(original.get('manifest')),
                json.dumps(original.get('raw')), original.get('hash'), client['first_seen'], client['last_seen'],
                json.dumps(client['stats']), json.dumps(client['cost'])))


def forget(identities):
    """Delete sessions from memory and the database. Their provider costs stay in the all-time total."""
    with LOCK:
        for identity in identities:
            CLIENTS.pop(identity, None)
            if DB is not None:
                DB.execute('DELETE FROM clients WHERE id = ?', (identity,))
    return len(identities)


def offline_clients(now=None):
    now = now or time.time()
    with LOCK:
        return [name for name, client in CLIENTS.items() if now - client['last_seen'] >= 15]


def mismatch(identity, message, source):
    """The game and its manifest disagree: a decide request the service rejected, or a mismatch the SDK reported."""
    with LOCK:
        client = get_client(identity)
        client['stats']['mismatches'] += 1
        client['mismatch'] = {'message': text(message, 300), 'at': time.time(), 'source': source}
        event(client, 'error', 'Manifest mismatch (%s): %s' % (source, text(message, 300)))


def set_setting(name, value):
    with LOCK:
        SETTINGS[name] = value
        if DB is not None:
            DB.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', (name, value))


def set_override(game, raw, manifest, digest):
    """Use a console-edited manifest (raw, validated, hash) for every session of `game`; None restores the game's own."""
    with LOCK:
        if raw is None:
            OVERRIDES.pop(game, None)
        else:
            OVERRIDES[game] = {'raw': raw, 'manifest': manifest, 'hash': digest, 'at': time.time()}
        if DB is not None:
            if raw is None:
                DB.execute('DELETE FROM overrides WHERE game = ?', (game,))
            else:
                DB.execute('INSERT OR REPLACE INTO overrides VALUES (?,?,?,?,?)', (game, json.dumps(raw), json.dumps(manifest), digest, OVERRIDES[game]['at']))
        for client in CLIENTS.values():
            if client['game'] == game and client.get('game_manifest'):
                apply_manifest(client)
                event(client, 'context', 'Manifest override ' + ('cleared: using the game manifest' if raw is None else 'applied from the console'))


def apply_manifest(client):
    """Effective manifest: the console override for this game if any, else what the game sent. Caller holds LOCK."""
    chosen = OVERRIDES.get(client['game']) or client['game_manifest']
    client['manifest'], client['manifest_hash'] = copy.deepcopy(chosen['manifest']), chosen['hash']


def event(client, kind, message, payload=None, entity=None):
    global SEQUENCE
    SEQUENCE += 1
    client['logs'].append({'id': SEQUENCE, 'at': time.time(), 'kind': kind, 'message': message, 'entity': entity, 'payload': payload})


def empty_stats():
    return {'batches': 0, 'provider_calls': 0, 'questions': 0, 'ai': 0, 'cached': 0, 'fallback': 0, 'fallback_codes': {},
            'pending': 0, 'reports': 0, 'reports_ignored': 0, 'context_updates': 0, 'latency_ms': None, 'mismatches': 0}


def get_client(identity, restored=False):
    if identity not in CLIENTS:
        now = time.time()
        for name in [name for name, client in CLIENTS.items() if now - client['last_seen'] > IDLE_DROP_S]:
            del CLIENTS[name]
        CLIENTS[identity] = {
            'id': identity, 'game': 'Unknown game', 'engine': '', 'protocol': None, 'profile': 'local',
            'manifest': None, 'manifest_hash': None, 'first_seen': now, 'last_seen': now, 'telemetry': {},
            'stats': empty_stats(), 'memory': {}, 'logs': deque(maxlen=MAX_LOGS), 'cost': empty_cost(),
            'population': {'scene': None, 'seen': {}}, 'context': {},
            'issued': {},        # decision_id -> {entity, action, reported}
            'last_decision': {},  # entity id -> {type, action, label, at}
            'links': {},         # entity id -> parent id (from decide requests and telemetry)
            'game_manifest': None,  # {raw, manifest, hash} as the game sent it
        }
        if not restored:
            event(CLIENTS[identity], 'connected', 'Client first seen')
    return CLIENTS[identity]


def session(identity, game, engine, protocol, manifest, manifest_hash, raw=None, warnings=()):
    with LOCK:
        client = get_client(identity)
        client.update({'game': game, 'engine': engine, 'protocol': protocol, 'last_seen': time.time(),
                       'game_manifest': {'raw': raw, 'manifest': manifest, 'hash': manifest_hash}})
        apply_manifest(client)
        event(client, 'session', f'Session started · {game} · {len(manifest["entity_types"])} entity types · {manifest_hash[:19]}…',
              {'entity_types': list(manifest['entity_types']), 'scopes': list(manifest['scopes']), 'profiles': list(manifest['profiles']), 'manifest_hash': manifest_hash})
        for warning in warnings:
            event(client, 'error', warning)
        if game in OVERRIDES:
            event(client, 'context', 'Manifest override active: the service uses the console-edited manifest, not the one the game sent')
        persist(client)


def manifest(identity):
    with LOCK:
        client = CLIENTS.get(identity)
        return client['manifest'] if client else None


def set_context(identity, scopes, profiles, changes):
    """Replace the session's scopes and profiles with an already validated result."""
    with LOCK:
        client = get_client(identity)
        client['manifest'] = dict(client['manifest'], scopes=scopes, profiles=profiles)
        client['stats']['context_updates'] += 1
        event(client, 'context', f'Context updated · {len(scopes)} scopes, {len(profiles)} profiles', changes)


def telemetry(data):
    identity = client_id(data.get('client_id'))
    mode = data.get('mode')
    if mode not in ('native', 'ai'):
        raise ValueError('mode must be native or ai')
    if not isinstance(data.get('paused'), bool):
        raise ValueError('Invalid paused flag')
    snapshot = {'mode': mode, 'paused': data['paused'], 'scene': text(data.get('scene', ''), 80),
                'last_error': text(data.get('last_error', '')), 'world': scalars(data.get('world'))}
    if data.get('scope') is not None:
        snapshot['scope'] = key(data['scope'], 40)
    entities = data.get('entities')
    if not isinstance(entities, list) or len(entities) > MAX_ENTITIES:
        raise ValueError('Expected at most %d entities' % MAX_ENTITIES)
    snapshot['entities'] = []
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError('Invalid entity')
        clean = {'id': key(entity.get('id')), 'type': key(entity.get('type'), 40), 'name': text(entity.get('name', ''), 80),
                 'state': text(entity.get('state', ''), 40), 'pending': entity.get('pending') is True,
                 'fields': scalars(entity.get('fields')), 'decision': scalars(entity.get('decision'), 12, lambda name: key(name, 40))}
        if entity.get('position') is not None:
            clean['position'] = position(entity['position'])
        for link in ('parent', 'profile'):
            if entity.get(link) is not None:
                clean[link] = key(entity[link], 80 if link == 'parent' else 40)
        snapshot['entities'].append(clean)
    with LOCK:
        client = get_client(identity)
        old = client['telemetry']
        if snapshot['last_error'].startswith('Manifest mismatch') and snapshot['last_error'] != (old or {}).get('last_error'):
            mismatch(identity, snapshot['last_error'][len('Manifest mismatch: '):], 'sdk')  # SDKs keep invalid requests to themselves and say so here
        if old and old.get('mode') != snapshot['mode']:
            client['memory'].clear()  # a mode switch drops in-flight decisions; start memory fresh
        for name in ('mode', 'scene', 'scope', 'paused', 'last_error'):
            if old.get(name) != snapshot.get(name):
                event(client, 'error' if name == 'last_error' and snapshot[name] else 'state', f'{name}: {snapshot.get(name)}')
        old_entities = {e['id']: e for e in old.get('entities', [])}
        for entity in snapshot['entities']:
            previous = old_entities.get(entity['id'])
            if previous is None or previous['state'] != entity['state']:
                event(client, 'execution', f"{entity['name'] or entity['id']}: {entity['state']}", entity, entity['id'])
            if 'parent' in entity:
                client['links'][entity['id']] = entity['parent']
        for gone in old_entities.keys() - {e['id'] for e in snapshot['entities']}:
            for store in ('memory', 'context', 'last_decision', 'links'):
                client[store].pop(gone, None)
            event(client, 'state', (old_entities[gone]['name'] or gone) + ': left scene or removed', entity=gone)
        population = client['population']
        if population['scene'] != snapshot['scene']:
            population.update({'scene': snapshot['scene'], 'seen': {}})
        for entity in snapshot['entities']:
            population['seen'].setdefault(entity['type'], set()).add(entity['id'])
        client['telemetry'] = snapshot
        client['last_seen'] = time.time()
        if client['last_seen'] - client.get('saved_at', 0) > 10:
            persist(client)  # ponytail: last_seen in the database lags up to 10 s; telemetry arrives every second
    return {'ok': True}


def population_summary(client):
    alive = {}
    for entity in client['telemetry'].get('entities', []):
        alive[entity['type']] = alive.get(entity['type'], 0) + 1
    return {kind: {'alive': alive.get(kind, 0), 'seen': len(ids), 'gone': len(ids) - alive.get(kind, 0)}
            for kind, ids in client['population']['seen'].items()}


def recall(identity, entity_id):
    with LOCK:
        client = CLIENTS.get(identity)
        return copy.deepcopy(list(client['memory'].get(entity_id, []))) if client else []


def situation(identity, entities, manifest):
    """Session-scoped context computed from this client's telemetry and the current batch:
    shared {population, observed}, and per entity {nearby, parent, members}."""
    # ponytail: O(n²) over ≤128 telemetry entities per batch; add a spatial index if scenes get much larger.
    now = time.time()
    with LOCK:
        client = CLIENTS.get(identity)
        known = copy.deepcopy(client['telemetry'].get('entities', [])) if client else []
        shared = {'population': population_summary(client) if client else {}, 'observed': []}
        last = copy.deepcopy(client['last_decision']) if client else {}
        links = dict(client['links']) if client else {}
        for entity in entities:
            if entity.get('parent') and client:
                client['links'][entity['id']] = links[entity['id']] = entity['parent']
    types = manifest['entity_types']
    by_id = {entity['id']: entity for entity in known}
    for entity in entities:  # the batch is fresher than telemetry
        merged = dict(by_id.get(entity['id'], {'id': entity['id'], 'type': entity['type'], 'state': '', 'decision': {}, 'fields': {}}))
        if entity.get('position'):
            merged['position'] = entity['position']
        by_id[entity['id']] = merged
    placed = [entity for entity in by_id.values() if entity.get('position')]

    def neighbours(origin, own_id, radius, pool):
        found = []
        for other in pool:
            if other['id'] == own_id or not other.get('position') or len(other['position']) != len(origin):
                continue
            distance = math.dist(origin, other['position'])
            if radius is None or distance <= radius:
                found.append((distance, other))
        return sorted(found, key=lambda pair: pair[0])

    def summary(other, distance=None):
        item = {'id': other['id'], 'type': other['type'], 'state': other.get('state', ''),
                'action': other.get('decision', {}).get('action') or last.get(other['id'], {}).get('action')}
        if distance is not None:
            item['distance'] = round(distance, 1)
        return dict(item, **other.get('fields', {}))

    for entity in placed:
        spec = types.get(entity['type'])
        if spec is not None and not (spec['actions'] or spec.get('decisions')):
            radius = spec.get('perception_radius')
            near = {}
            for _, other in neighbours(entity['position'], entity['id'], radius, placed) if radius else []:
                near[other['type']] = near.get(other['type'], 0) + 1
            shared['observed'].append({'id': entity['id'], 'type': entity['type'], 'fields': entity.get('fields', {}), 'nearby_counts': near})

    per_entity = {}
    for entity in entities:
        context = {}
        me = by_id[entity['id']]
        radius = types[entity['type']].get('perception_radius')
        if radius and me.get('position'):
            context['nearby'] = [summary(other, distance) for distance, other in neighbours(me['position'], entity['id'], radius, placed)[:5]]
        parent_id = entity.get('parent') or links.get(entity['id'])
        if parent_id:
            decision = last.get(parent_id, {})
            context['parent'] = {'id': parent_id, 'type': decision.get('type') or by_id.get(parent_id, {}).get('type'),
                                 'action': decision.get('action'), 'label': decision.get('label'),
                                 'seconds_ago': round(now - decision['at'], 1) if decision.get('at') else None}
        members = [by_id[child] for child, parent in links.items() if parent == entity['id'] and child in by_id]
        if members:
            if me.get('position'):
                ranked = neighbours(me['position'], entity['id'], None, members)
                rest = [m for m in members if not m.get('position')]
                members = [m for _, m in ranked] + rest
            context['members'] = [summary(member) for member in members[:8]]
        per_entity[entity['id']] = context
    with LOCK:
        if identity in CLIENTS:
            for entity in entities:
                CLIENTS[identity]['context'][entity['id']] = dict(per_entity[entity['id']], at=now, population=shared['population'],
                                                                   trigger=entity.get('trigger'), profile=entity.get('profile'))
    return shared, per_entity


def note_tokens(identity, entity_id, tokens, budget):
    with LOCK:
        client = CLIENTS.get(identity)
        if client and entity_id in client['context']:
            client['context'][entity_id].update({'input_tokens': tokens, 'max_input_tokens': budget})


def request_started(identity, entities):
    with LOCK:
        client = get_client(identity)
        client['last_seen'] = time.time()
        client['stats']['batches'] += 1
        client['stats']['questions'] += len(entities)
        client['stats']['pending'] += 1
        event(client, 'request', f'Decision batch · {len(entities)} entities', {'entities': entities})


def action_text(action):
    """One action, or "slot=action, ..." for a decision with slots (as decided, or as a report's executed actions)."""
    if isinstance(action, dict):
        return ', '.join('%s=%s' % (slot, item.get('action') or item.get('code') if isinstance(item, dict) else item) for slot, item in action.items())
    return action


def request_finished(identity, decisions, latency_ms, error=None, cost=None, provider_called=False):
    with LOCK:
        client = get_client(identity)
        stats = client['stats']
        stats['pending'] = max(0, stats['pending'] - 1)
        stats['latency_ms'] = latency_ms
        if provider_called:
            stats['provider_calls'] += 1
        if cost:
            add_cost(client['cost'], cost)
            add_cost(TOTAL_COST, cost)
            if DB is not None:
                DB.execute('INSERT INTO costs VALUES (?,?,?,?,?,?,?,?)', (time.time(), identity, client['game'], cost['provider'],
                           cost['input_tokens'], cost['output_tokens'], cost['usd'], int(cost['priced'])))
        if error:
            event(client, 'error', 'Decision batch failed: ' + error)
        now = time.time()
        for decision in decisions:
            ai = decision['source'] == 'AI'
            if ai:
                stats['ai'] += 1
                if decision.get('cached'):
                    stats['cached'] += 1
            else:
                stats['fallback'] += 1
                stats['fallback_codes'][decision['code']] = stats['fallback_codes'].get(decision['code'], 0) + 1
            action = action_text(decision.get('action') or decision.get('slots'))
            message = f"{decision['id']} → {action or decision.get('code')} · {decision['source']}" + (' · cached' if decision.get('cached') else '')
            event(client, 'decision' if ai else 'error', message, {k: v for k, v in decision.items() if k != 'legal'}, decision['id'])
            if ai:
                if len(client['issued']) >= MAX_ISSUED:
                    client['issued'].pop(next(iter(client['issued'])))
                # Any legal action may come back in the report: a game can sample from the probabilities.
                client['issued'][decision['decision_id']] = {'entity': decision['id'], 'legal': decision.get('legal'), 'reported': False}
                kind = decision.get('type')
                client['last_decision'][decision['id']] = {'type': kind, 'action': action, 'label': decision.get('label'), 'at': now}
                memory = client['memory'].setdefault(decision['id'], deque(maxlen=MEMORY_SIZE))
                memory.append({'decision_id': decision['decision_id'], 'at': now, 'action': action, 'label': decision.get('label'),
                               'confidence': decision.get('confidence'), 'cached': decision.get('cached', False),
                               'trigger': decision.get('trigger'), 'status': 'pending', 'outcome': {}})
        persist(client)


def report(identity, entity_id, decision_id, action, status, outcome):
    """Apply a report only for a decision issued to this session and entity, once, with actions that were legal for it.
    `action` is one action, or {slot: action} for a decision with slots. Returns True if applied."""
    with LOCK:
        client = get_client(identity)
        issued = client['issued'].get(decision_id)
        legal = issued and issued['legal']
        if isinstance(legal, dict):
            fits = isinstance(action, dict) and action and all(slot in legal and value in legal[slot] for slot, value in action.items())
        else:
            fits = isinstance(action, str) and action in (legal or [])
        if issued is None or issued['entity'] != entity_id or not fits or issued['reported']:
            client['stats']['reports_ignored'] += 1
            event(client, 'error', f'{entity_id}: report ignored (unknown, foreign or repeated decision_id)', {'decision_id': decision_id}, entity_id)
            return False
        issued['reported'] = True
        client['stats']['reports'] += 1
        for item in client['memory'].get(entity_id, []):
            if item['decision_id'] == decision_id:  # remember what the game actually executed
                item.update({'action': action_text(action), 'status': status, 'outcome': outcome})
        event(client, 'report', f'{entity_id}: {action_text(action)} {status}', {'decision_id': decision_id, 'action': action, 'status': status, 'outcome': outcome}, entity_id)
        persist(client)
        return True


def snapshot(identity=None):
    with LOCK:
        output = []
        now = time.time()
        hidden = ('logs', 'memory', 'telemetry', 'manifest', 'population', 'context', 'issued', 'last_decision', 'links', 'game_manifest', 'saved_at')
        for name, client in CLIENTS.items():
            if identity is not None and name != identity:
                continue
            data = {field: copy.deepcopy(value) for field, value in client.items() if field not in hidden}
            data['age_seconds'] = round(now - client['last_seen'], 1)
            data['status'] = 'online' if data['age_seconds'] < 5 else ('stale' if data['age_seconds'] < 15 else 'offline')
            data['scene'] = client['telemetry'].get('scene', 'Awaiting telemetry')
            data['scope'] = client['telemetry'].get('scope')
            data['mode'] = client['telemetry'].get('mode', 'Unknown')
            data['entity_count'] = len(client['telemetry'].get('entities', []))
            data['population'] = population_summary(client)
            override = OVERRIDES.get(client['game']) if client.get('game_manifest') else None
            data['override'] = {'at': override['at'], 'manifest_hash': override['hash']} if override else None
            if identity is not None:
                data['game_manifest'] = copy.deepcopy(client['game_manifest'])
                data['manifest_raw'] = copy.deepcopy(override['raw'] if override else (client['game_manifest'] or {}).get('raw'))
                data['telemetry'] = copy.deepcopy(client['telemetry'])
                data['manifest'] = copy.deepcopy(client['manifest'])
                data['memory'] = {entity: list(copy.deepcopy(items)) for entity, items in client['memory'].items()}
                data['context'] = copy.deepcopy(client['context'])
                data['links'] = dict(client['links'])
                data['logs'] = list(copy.deepcopy(client['logs']))
            output.append(data)
        if identity is not None:
            return output[0] if output else None
        return {'clients': output, 'uptime_seconds': round(now - STARTED), 'cost': copy.deepcopy(TOTAL_COST), 'persistent': DB is not None}
