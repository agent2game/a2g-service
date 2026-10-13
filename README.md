# A2G reference decision service

The reference decision service for the [Agent2Game (A2G) protocol](https://agent2game.org) 0.1. A game's A2G client (for example the [Godot](https://github.com/agent2game/a2g-sdk-godot) or [JS/TS](https://github.com/agent2game/a2g-sdk-js) SDK) sends it a manifest, the situation and each entity's legal actions; the service asks a decision provider (OpenAI Decisions or TypeSafe Jev) and returns one validated choice per entity, or a fallback so the game's native AI keeps playing. It also serves the **Agent2Game Console**, a browser view of sessions, entities, decisions, costs and logs.

The service holds the provider key; it is never sent to the game or the console. It is game-agnostic: everything game-specific comes from the manifest.

| Service | Protocol | Runtime |
| --- | --- | --- |
| `0.1.2` | `0.1` | Docker, or Python 3.12+ |

## Run

With Docker, using the published image (amd64 and arm64):

```sh
docker run -d --name a2g -p 127.0.0.1:8787:8787 -v a2g-data:/data -e TYPESAFE_API_KEY=... -e A2G_PROVIDER=jev ghcr.io/agent2game/a2g-service:0.1.2
```

Or from this repository: `cp .env.example .env`, put a provider key in it, then `docker compose up -d --build`. Without Docker: `pip install -r requirements.txt`, then `python3 decision_service.py`.

Open http://127.0.0.1:8787/ for the Agent2Game Console and http://127.0.0.1:8787/health for status.

**Keep `-p 127.0.0.1:...`.** The service runs the `local` [security profile](https://agent2game.org/specification/security/#deployment-profiles): no authentication, meant for the developer's machine. Publishing the port on all interfaces (`-p 8787:8787`) would expose it, and your provider key's budget, to your network.

## Configuration

Environment variables (in Docker with `-e`, or in `.env`; the process environment wins):

| Variable | Meaning |
| --- | --- |
| `OPENAI_API_KEY`, `TYPESAFE_API_KEY` | Provider keys; only the active provider's key is needed |
| `A2G_PROVIDER` | `openai` (default) or `jev`; a choice made in the Agent2Game Console overrides it |
| `A2G_ALLOWED_ORIGINS` | Exact loopback origins of browser games on local dev servers, comma-separated, for example `http://localhost:5173` |
| `A2G_CONSOLE` | `off` serves only `/v1/*` and `/health` |
| `A2G_PROVIDER_CALLS` | Concurrent provider calls (default 16) |
| `A2G_QUESTIONS_PER_CALL` | Questions per provider call (default 64); bigger batches, for example entities with many decision slots, are split into parallel calls |
| `A2G_PORT`, `A2G_DB` | Listen port (default 8787) and SQLite file (Docker: `/data/a2g.db` in the volume) |

What it implements, its limits and the status of every protocol feature: [Reference service](https://agent2game.org/implementations/service/).

## Tests

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
```

The tests mock the provider; they make no paid calls. With a sibling checkout of [a2g-protocol](https://github.com/agent2game/a2g-protocol) (`npm ci` in its `schema/`), they also check the service against the protocol's JSON Schema and examples.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
