## Playful chef api

Run raw python with live reload:

```sh
poetry run uvicorn playful_chef_api.main:app --reload --host 0.0.0.0 --port 8000
```

Run in docker:

```sh
docker-compose up --build
```

Prepare data:

```sh
poetry run python3 data/csv_to_sqlite.py
```
