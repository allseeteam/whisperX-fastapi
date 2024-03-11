### This directory codebase contains FastAPI implementation for WhisperX with Docker .

### Installation

#### Prerequisites

- Download the repository
```bash
git clone https://github.com/allseeteam/whisperx-fastapi
cd whisperx-fastapi
```

- GPU execution requires the NVIDIA libraries cuBLAS 11.x and cuDNN 8.x to be installed on the system. Please refer to the CTranslate2 documentation.

- Virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Running FastAPI locally (GPU)
```bash
pip install -r fastapi/requirements-fastapi-cuda.txt
pip install -e .
cd fastapi
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

#### Running FastAPI with Docker (GPU)
```bash
sudo docker build -f fastapi/dockerization/dockerfile.fastapi.cuda -t whisperx-fastapi-cuda .
sudo docker run -p 8000:8000 --env-file ./fastapi/env/.env.cuda  --gpus all --name whisperx-fastapi-cuda-container whisperx-fastapi-cuda
```

### *Currently only supports GPU. If someone interested in adding support for CPU or any other contributions, we would be happy to accept your PRs 😎*