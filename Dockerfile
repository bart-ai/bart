FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY ./src ./src

RUN pip3 install -r ./src/requirements.txt

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

WORKDIR "./src"
ENTRYPOINT ["streamlit", "run", "./web.py", "--server.port=8080", "--server.address=0.0.0.0"]
