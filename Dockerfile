FROM python:3.11-slim

WORKDIR /app

# Copy everything first
COPY . .

# Install dependencies - openenv-core may not be on PyPI, ignore if missing
RUN pip install --no-cache-dir fastapi uvicorn pydantic openai pyyaml || true
RUN pip install --no-cache-dir openenv-core || true

# Verify app.py is present and importable
RUN python -c "import app; print('app.py import OK')"

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]