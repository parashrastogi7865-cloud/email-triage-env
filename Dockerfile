FROM python:3.11-slim

WORKDIR /app

# Copy all files into the container
COPY . .

# Install necessary libraries
RUN pip install --no-cache-dir fastapi uvicorn pydantic openai pyyaml openenv-core

# Set the PYTHONPATH so the server folder is recognized as a package
ENV PYTHONPATH=/app

# Updated verification command to look inside the server folder
RUN python -c "from server import app; print('app.py import OK')"

EXPOSE 7860

# Start the server using the module path
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]