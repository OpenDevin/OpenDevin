FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y python3 python3-pip git gcc

WORKDIR /root

COPY requirements.txt .
RUN pip install -r requirements.txt

# docker build -t xingyaoww/od-eval-mint:v1.0 .
