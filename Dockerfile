FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /app

# install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN ["pip", "install", "--no-cache-dir", "--upgrade", "-r", "/app/requirements.txt"]
# install dependency on top of base image torch
RUN ["pip", "install", "--no-cache-dir", "--upgrade", "sentence-transformers==5.2.0"]

COPY ./playful_chef_api /app/playful_chef_api
COPY ./data/database.db /app/data/database.db
COPY ./index/faiss_index/ /app/index/faiss_index/

EXPOSE 8000
CMD ["uvicorn", "playful_chef_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
