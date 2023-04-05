FROM python:3.10

RUN apt-get update
RUN apt-get -y install software-properties-common
RUN apt-get -y install git curl vim bash sudo tmux ack less man jq
RUN apt-get -y install graphviz
# RUN apt-get -y install default-jdk
# RUN apt-get -y install php php-ast composer

RUN pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN useradd -c '' -m -d /home/u -G sudo -s /bin/bash u
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER u
WORKDIR /home/u

CMD ["/bin/bash"]
