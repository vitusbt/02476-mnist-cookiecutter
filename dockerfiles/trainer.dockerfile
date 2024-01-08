# Base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mnist_exercise/ mnist_exercise/
COPY data/ data/

WORKDIR /
RUN pip install . --no-cache-dir --default-timeout=100 #(1)

ENV PYTHONPATH "/"

ENTRYPOINT ["python", "-u", "mnist_exercise/train_model.py"]
