## Playful chef api

Prepare project (requires [poetry](https://python-poetry.org/docs/#installation)):

```sh
poetry install
poetry run pre-commit install
```

Prepare data: pull submodule, then convert to SQLite:

```sh
git submodule update --init --recursive
poetry run python3 data/csv_to_sqlite.py
poetry run python index/index_builder.py
```

Run raw python with live reload:

```sh
poetry run uvicorn playful_chef_api.main:app --reload --host 0.0.0.0 --port 8000
```

Agent endpoint `/agent` returns a structured recipe (full data). It requires `LLM_API_KEY` (Mistral):
```sh
export LLM_API_KEY=...   # or $env:LLM_API_KEY=... on PowerShell
poetry run uvicorn playful_chef_api.main:app --reload --host 0.0.0.0 --port 8000
```
Without the key, `/recipes` and `/ingredients` still work; `/agent` responds 503.

Run in podman:

```sh
poetry run podman-compose up --build -d
```
