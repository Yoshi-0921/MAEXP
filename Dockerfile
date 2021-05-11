FROM python:3.9.4

ENV HOME /home
WORKDIR $HOME

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git

RUN git clone https://github.com/Yoshi-0921/MAEXP

RUN pip install -r MAEXP/requirements.txt
