FROM python:3.12-slim-trixie

WORKDIR /app

# install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN ["pip", "install", "--no-cache-dir", "--upgrade", "-r", "/app/requirements.txt"]

# app code and data/scripts
COPY ./playful_chef_api /app/playful_chef_api
COPY ./data /app/data
COPY ./index /app/index

# build sqlite db and faiss index inside the image
RUN python /app/data/csv_to_sqlite.py
RUN python /app/index/index_builder.py

EXPOSE 8000
CMD ["uvicorn", "playful_chef_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
