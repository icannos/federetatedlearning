FROM tensorflow/tensorflow:latest-gpu-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y --no-install-recommends python3-opencv
RUN pip3 install pandas numpy scipy matplotlib keras


COPY . app/
