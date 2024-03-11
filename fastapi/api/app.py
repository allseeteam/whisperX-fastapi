# Copyright (C) 2024 Gregory Matsnev
#
# This file is part of the modifications made to the WhisperX project by Gregory Matsnev.


import asyncio
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from threading import Thread
from typing import Any

from fastapi import HTTPException, BackgroundTasks, FastAPI, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from streaming_form_data import StreamingFormDataParser
from streaming_form_data.targets import FileTarget
from streaming_form_data.validators import MaxSizeValidator

import whisperx


@dataclass
class WhisperXModels:
    whisper_model: Any
    diarize_pipeline: Any
    align_model: Any
    align_model_metadata: Any


class TranscriptionAPISettings(BaseSettings):
    tmp_dir: str = 'tmp'
    cors_origins: str = '*'
    cors_allow_credentials: bool = True
    cors_allow_methods: str = '*'
    cors_allow_headers: str = '*'
    whisper_model: str = 'large-v2'
    device: str = 'cuda'
    compute_type: str = 'float16'
    batch_size: int = 16
    language_code: str = 'auto'
    hf_api_key: str = ''
    file_loading_chunk_size_mb: int = 1024
    task_cleanup_delay_min: int = 60
    max_file_size_mb: int = 4096
    max_request_body_size_mb: int = 5000

    class Config:
        env_file = 'env/.env.cuda'
        env_file_encoding = 'utf-8'


class MaxBodySizeException(Exception):
    def __init__(self, body_len: int):
        self.body_len = body_len


class MaxBodySizeValidator:
    def __init__(self, max_size: int):
        self.body_len = 0
        self.max_size = max_size

    def __call__(self, chunk: bytes):
        self.body_len += len(chunk)
        if self.body_len > self.max_size:
            raise MaxBodySizeException(self.body_len)


settings = TranscriptionAPISettings()

app = FastAPI()
# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(','),
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods.split(','),
    allow_headers=settings.cors_allow_headers.split(','),
)

trancription_tasks = {}
trancription_tasks_queue = Queue()

whisperx_models = WhisperXModels(
    whisper_model=None,
    diarize_pipeline=None,
    align_model=None,
    align_model_metadata=None
)


def load_whisperx_models() -> None:
    global whisperx_models

    whisperx_models.whisper_model = whisperx.load_model(
        whisper_arch=settings.whisper_model,
        device=settings.device,
        compute_type=settings.compute_type,
        language=settings.language_code if settings.language_code != "auto" else None
    )

    whisperx_models.diarize_pipeline = whisperx.DiarizationPipeline(
        use_auth_token=settings.hf_api_key,
        device=settings.device
    )

    if settings.language_code != "auto":
        (
            whisperx_models.align_model,
            whisperx_models.align_model_metadata
        ) = whisperx.load_align_model(
            language_code=settings.language_code,
            device=settings.device
        )


def transcribe_audio(audio_file_path: str) -> dict:
    global whisperx_models

    audio = whisperx.load_audio(audio_file_path)

    transcription_result = whisperx_models.whisper_model.transcribe(
        audio,
        batch_size=int(settings.batch_size),
    )

    if settings.language_code == "auto":
        language = transcription_result["language"]
        (
            whisperx_models.align_model,
            whisperx_models.align_model_metadata
        ) = whisperx.load_align_model(
            language_code=language,
            device=settings.device
        )

    aligned_result = whisperx.align(
        transcription_result["segments"],
        whisperx_models.align_model,
        whisperx_models.align_model_metadata,
        audio,
        settings.device,
        return_char_alignments=False
    )

    diarize_segments = whisperx_models.diarize_pipeline(audio)

    final_result = whisperx.assign_word_speakers(
        diarize_segments,
        aligned_result
    )

    return final_result


def transcription_worker() -> None:
    while True:
        task_id, tmp_path = trancription_tasks_queue.get()

        try:
            result = transcribe_audio(tmp_path)
            trancription_tasks[task_id].update({"status": "completed", "result": result})

        except Exception as e:
            trancription_tasks[task_id].update({"status": "failed", "result": str(e)})

        finally:
            trancription_tasks_queue.task_done()
            os.remove(tmp_path)


@app.on_event("startup")
async def startup_event() -> None:
    os.makedirs(settings.tmp_dir, exist_ok=True)
    load_whisperx_models()
    Thread(target=transcription_worker, daemon=True).start()


async def cleanup_task(task_id: str) -> None:
    await asyncio.sleep(settings.task_cleanup_delay_min * 60)
    trancription_tasks.pop(task_id, None)


@app.post("/transcribe/")
async def create_upload_file(
        request: Request,
        background_tasks: BackgroundTasks
) -> dict:
    task_id = str(uuid.uuid4())
    tmp_path = f"{settings.tmp_dir}/{task_id}.audio"

    trancription_tasks[task_id] = {
        "status": "loading",
        "creation_time": datetime.utcnow(),
        "result": None
    }

    body_validator = MaxBodySizeValidator(settings.max_request_body_size_mb * 1024 * 1024)

    try:
        file_target = FileTarget(
            tmp_path,
            validator=MaxSizeValidator(settings.max_file_size_mb * 1024 * 1024)
        )
        parser = StreamingFormDataParser(headers=request.headers)
        parser.register('file', file_target)
        async for chunk in request.stream():
            body_validator(chunk)
            parser.data_received(chunk)

    except MaxBodySizeException as e:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Maximum request body size limit exceeded: {e.body_len} bytes"
        )

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing upload: {str(e)}"
        )

    if not file_target.multipart_filename:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='No file was uploaded'
        )

    trancription_tasks[task_id].update({"status": "processing"})
    trancription_tasks_queue.put((task_id, tmp_path))

    background_tasks.add_task(cleanup_task, task_id)

    return {
        "task_id": task_id,
        "creation_time": trancription_tasks[task_id]["creation_time"].isoformat(),
        "status": trancription_tasks[task_id]["status"]
    }


@app.get("/transcribe/status/{task_id}")
async def get_task_status(task_id: str) -> dict:
    task = trancription_tasks.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "creation_time": task["creation_time"],
        "status": task["status"]
    }


@app.get("/transcribe/result/{task_id}")
async def get_task_result(task_id: str) -> dict:
    task = trancription_tasks.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] == "pending":
        raise HTTPException(status_code=404, detail="Task not completed")

    return {
            "task_id": task_id,
            "creation_time": task["creation_time"],
            "status": task["status"],
            "result": task["result"]
    }
