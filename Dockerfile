FROM --platform=linux/amd64 python:3.9-alpine
WORKDIR /app
# Update repos
RUN echo "http://dl-cdn.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories
RUN echo "http://dl-cdn.alpinelinux.org/alpine/edge/main" >> /etc/apk/repositories
RUN apk update
RUN apk --no-cache --update add \
    musl-dev linux-headers g++ openblas-dev libffi-dev \
    geos-dev alpine-sdk gcc chromium chromium-chromedriver
ENV PATH="/usr/bin/chromedriver:${PATH}"
# Make it a separate layer to cache long compilation
RUN pip install scipy numpy
RUN pip install --upgrade pip
COPY requirements.txt .
# Install dependencies in parallel
RUN xargs -n 1 -P 8 pip install < requirements.txt
COPY . .

ENTRYPOINT [ "python", "main.py" ]
