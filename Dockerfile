FROM python:3.14-alpine

WORKDIR /app

# install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN ["pip", "install", "--no-cache-dir", "--upgrade", "-r", "/app/requirements.txt"]

COPY ./playful_chef_api /app/playful_chef_api
COPY ./data/database.db /app/data/database.db

EXPOSE 80
CMD ["uvicorn", "playful_chef_api.main:app", "--host", "0.0.0.0", "--port", "80"]
