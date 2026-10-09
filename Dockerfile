FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY riskplatform ./riskplatform
COPY sql ./sql
COPY dashboard ./dashboard
ENV RISK_ARTIFACT_DIR=/app/artifacts RISK_DATA_DIR=/app/data
RUN useradd -m app && mkdir -p /app/artifacts /app/data && chown -R app /app
USER app
EXPOSE 8000 8501
CMD ["uvicorn", "riskplatform.api:app", "--host", "0.0.0.0", "--port", "8000"]
