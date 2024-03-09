# Copyright (C) 2024 Gregory Matsnev
#
# This file is part of the modifications made to the WhisperX project by Gregory Matsnev.


FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

ARG HOST=0.0.0.0
ARG PORT=8000

ENV HOST=${HOST}
ENV PORT=${PORT}

CMD uvicorn api:app --host ${HOST} --port ${PORT}
