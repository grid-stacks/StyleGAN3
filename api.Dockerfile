# pull the official docker image
#FROM python:3.9-slim-bullseye

FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# Install apt packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    # dependencies for building Python packages
    # build-essential  \
    software-properties-common

RUN apt-get update -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install python3.9 -y
RUN apt-get install python3-pip -y

#RUN add-apt-repository ppa:graphics-drivers/ppa
#
#RUN ubuntu-drivers devices
#
#RUN ubuntu-drivers autoinstall

# set work directory
WORKDIR /app

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy project
COPY . .

RUN nvcc --version