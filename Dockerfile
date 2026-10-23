
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=7860

COPY pyproject.toml ./
COPY src/ ./src/

# RUN python -m pip install --upgrade pip && pip install --no-cache-dir .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --retries 10 --timeout 120 .


EXPOSE 7860

CMD ["sh", "-c", "python -m src.app"]
