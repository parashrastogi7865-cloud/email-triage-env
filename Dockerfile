FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . .

# Expose port for HF Spaces
EXPOSE 7860

# Default: run validation then launch demo API
CMD ["python", "app.py"]
