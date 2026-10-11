# Inference-only image: ONNX Runtime + OpenCV + FastAPI. No PyTorch, no training data.
# Build after `moodtune export` has produced models/moodtune.onnx:
#   docker build -t moodtune .
#   docker run -p 8000:8000 moodtune
# Browsers only allow camera access on localhost or HTTPS: put a TLS proxy in front when
# serving to other machines. Photo upload works either way.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir \
        "numpy>=2.0" "opencv-python-headless>=4.10" "onnxruntime>=1.18" \
        "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9" \
        "pydantic>=2.7" "pyyaml>=6.0"

COPY src/moodtune/__init__.py src/moodtune/config.py src/moodtune/labels.py \
     src/moodtune/preprocess.py src/moodtune/predictor.py src/moodtune/face.py \
     src/moodtune/analyzer.py src/moodtune/smoothing.py src/moodtune/music.py \
     /app/moodtune/
COPY src/moodtune/api /app/moodtune/api
COPY scripts/serve.py /app/serve.py
COPY configs /app/configs
COPY models/moodtune.onnx models/face_detection_yunet_2023mar.onnx /app/models/

# Read-only app: nothing is written at runtime, so the user owns nothing.
RUN useradd --no-create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["python", "serve.py", "--config", "configs/default.yaml", "--host", "0.0.0.0", "--port", "8000"]
