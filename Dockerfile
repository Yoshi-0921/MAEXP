FROM ubuntu:20.04

ENV HOME /home
WORKDIR $HOME

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git

RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools

# COPY ./configs /home/MAEXP/configs
# COPY ./core /home/MAEXP/core
# COPY ./main.py /home/MAEXP
# COPY ./requirements.txt /home/MAEXP

ENV PYTHON_VERSION 3.9.4
ENV PYTHONPATH "${PYTHONPATH}:/usr/bin/python3"

RUN git clone https://github.com/Yoshi-0921/MAEXP

RUN pip3 install -r MAEXP/requirements.txt
