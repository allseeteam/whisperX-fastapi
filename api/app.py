# Copyright (C) 2024 Gregory Matsnev
#
# This file is part of the modifications made to the WhisperX project by Gregory Matsnev.


import os
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException

import whisperx


@dataclass
class WhisperXModels:
    whisper_model: Any
    diarize_pipeline: Any
    align_model: Any
    align_model_metadata: Any


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        'CORS_ORIGINS',
        '*'
    ).split(','),
    allow_credentials=os.environ.get(
        'CORS_ALLOW_CREDENTIALS', 'true'
    ) == 'true',
    allow_methods=os.environ.get(
        'CORS_ALLOW_METHODS',
        '*'
    ).split(','),
    allow_headers=os.environ.get(
        'CORS_ALLOW_HEADERS',
        '*'
    ).split(','),
)

tasks = {}

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "large-v2"
)
DEVICE = os.getenv(
    "DEVICE",
    "cuda"
)
COMPUTE_TYPE = os.getenv(
    "COMPUTE_TYPE",
    "float16"
)
BATCH_SIZE = os.getenv(
    "BATCH_SIZE",
    16
)
LANGUAGE_CODE = os.getenv(
    "LANGUAGE_CODE",
    "auto"
)
HF_API_KEY = os.getenv(
    "HF_API_KEY",
    ''
)

whisperx_models = WhisperXModels(
    whisper_model=None,
    diarize_pipeline=None,
    align_model=None,
    align_model_metadata=None
)


def load_models():
    global whisperx_models
    whisperx_models.whisper_model = whisperx.load_model(
        whisper_arch=WHISPER_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        language=LANGUAGE_CODE if LANGUAGE_CODE != "auto" else None
    )
    whisperx_models.diarize_pipeline = whisperx.DiarizationPipeline(
        use_auth_token=HF_API_KEY,
        device=DEVICE
    )
    if LANGUAGE_CODE != "auto":
        (
            whisperx_models.align_model,
            whisperx_models.align_model_metadata
        ) = whisperx.load_align_model(
            language_code=LANGUAGE_CODE,
            device=DEVICE
        )


@app.on_event("startup")
async def startup_event():
    load_models()


async def transcribe_audio(audio_file_path):
    global whisperx_models

    audio = whisperx.load_audio(audio_file_path)

    transcription_result = model.transcribe(
        audio,
        batch_size=int(BATCH_SIZE),
    )

    if LANGUAGE_CODE == "auto":
        language = transcription_result["language"]
        (
            whisperx_models.align_model,
            whisperx_models.align_model_metadata
        ) = whisperx.load_align_model(
            language_code=language,
            device=DEVICE
        )

    aligned_result = whisperx.align(
        transcription_result["segments"],
        whisperx_models.align_model,
        whisperx_models.align_model_metadata,
        audio,
        DEVICE,
        return_char_alignments=False
    )

    diarize_segments = whisperx_models.diarize_pipeline(audio)

    final_result = whisperx.assign_word_speakers(
        diarize_segments,
        aligned_result
    )

    return final_result


async def transcribe_audio_wrapper(tmp_path, task_id):
    try:
        result = await transcribe_audio(tmp_path)
        tasks[task_id] = {"status": "completed", "result": result}
    except Exception as e:
        tasks[task_id] = {"status": "failed", "result": str(e)}


@app.post("/transcribe/")
async def create_upload_file(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...)
) -> dict:
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    task_id = str(uuid.uuid4())
    tmp_path = f"{tmp_dir}/{task_id}"

    with open(tmp_path, "wb") as buffer:
        buffer.write(await file.read())

    tasks[task_id] = {"status": "pending", "result": None}
    background_tasks.add_task(transcribe_audio_wrapper, tmp_path, task_id)

    return {"task_id": task_id, "detail": "Processing started."}


@app.get("/transcribe/status/{task_id}")
async def get_task_status(task_id: str) -> dict:
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "status": task['status']}


@app.get("/transcribe/result/{task_id}")
async def get_task_result(task_id: str) -> dict:
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task['status'] != 'completed':
        return {"task_id": task_id, "status": task['status'], "result": "Task not completed yet"}
    return {"task_id": task_id, "result": task['result']}
