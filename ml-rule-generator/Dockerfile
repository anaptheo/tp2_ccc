FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generate_rules.py .

ENTRYPOINT ["python", "generate_rules.py"]