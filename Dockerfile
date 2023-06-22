FROM nvcr.io/nvidia/pytorch:22.12-py3

ARG USER_ID
ARG GROUP_ID
ARG USER
ARG DEBIAN_FRONTEND=noninteractive

RUN addgroup --gid $GROUP_ID spinnawala
RUN adduser --disabled-password --gecos "" --uid $USER_ID --gid $GROUP_ID spinnawala

WORKDIR /nfs/home/$USER

COPY requirements.txt .
COPY requirements-dev.txt .

RUN pip install --upgrade pip
RUN apt-get update \
&& apt-get install -y sudo
RUN pip3 install -r requirements.txt
RUN pip3 install -r requirements-dev.txt

USER root
