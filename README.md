## Playful chef api

Prepare project (requires [poetry](https://python-poetry.org/docs/#installation)):

```sh
poetry install
poetry run pre-commit install
```

Prepare data: download and extract [kaggle dataset](https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m), then convert to SQLite:

```sh
cp "<extracted csv path>" ./data/recipes_data.csv
poetry run python3 data/csv_to_sqlite.py
```

Run raw python with live reload:

```sh
poetry run uvicorn playful_chef_api.main:app --reload --host 0.0.0.0 --port 8000
```

Run in docker:

```sh
docker-compose up --build
```

Run in podman:

```sh
podman build -t playful_chef_api:latest .
podman run -p 8000:80 -it playful_chef_api:latest
```
