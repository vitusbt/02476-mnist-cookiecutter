# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mnist-exercise/ mnist-exercise/
COPY data/ data/

WORKDIR /
RUN pip install . --no-cache-dir #(1)

ENTRYPOINT ["python", "-u", "mnist-exercise/models/train_model.py"]