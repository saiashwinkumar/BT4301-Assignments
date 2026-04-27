FROM python:3.12-slim

WORKDIR /app

# Copy requirement and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the files
COPY app.py ./

EXPOSE 8000

# Start fastapi
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
