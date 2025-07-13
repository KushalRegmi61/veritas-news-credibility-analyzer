FROM python:3.13.1-slim

# Set environment variables
ENV NLTK_DATA=/app/nltk_data \
    MODEL_PATH=/app/model/model_pipeline.pkl

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK resources during build
RUN python utils/nltk_setup.py

# Expose port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
