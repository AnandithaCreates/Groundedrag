FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# Only the base deps are needed at runtime -- ragas/langchain-community/
# langchain-groq are eval-only tools that never run inside the deployed
# service, so we skip them here to keep the image small.
RUN pip install --no-cache-dir \
    fastapi==0.115.0 "uvicorn[standard]==0.32.0" langgraph==0.2.45 \
    sentence-transformers==3.2.1 numpy==1.26.4 python-dotenv==1.0.1 \
    pydantic==2.9.2 qdrant-client==1.12.1 portkey-ai==1.10.0

COPY app/ ./app/
COPY static/ ./static/

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
