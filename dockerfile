# Copyright (C) 2024 Gregory Matsnev
#
# This file is part of the modifications made to the WhisperX project by Gregory Matsnev.


FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3-pip ffmpeg libsm6 libxext6 git

WORKDIR /whisperx-fastapi
COPY . .

RUN pip install -e .
RUN pip install fastapi uvicorn

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
