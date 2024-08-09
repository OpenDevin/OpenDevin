FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y python3 python3-pip git

RUN git clone https://github.com/Farama-Foundation/miniwob-plusplus.git /miniwob-plusplus && \
    git -C "/miniwob-plusplus" reset --hard 7fd85d71a4b60325c6585396ec4f48377d049838

ENV MINIWOB_URL="file:///miniwob-plusplus/miniwob/html/miniwob/"

# docker build -t xingyaoww/od-eval-miniwob .
