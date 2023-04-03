FROM python:3.10

RUN apt-get update
RUN apt-get -y install software-properties-common
RUN apt-get -y install git curl vim bash sudo tmux ack less man jq

RUN apt-get -y install default-jdk

RUN apt-get -y install php php-ast composer

RUN pip install --upgrade pip
RUN pip install typeguard==2.13.3
RUN pip install tqdm
RUN pip install joblib
RUN pip install numpy
RUN pip install scikit-learn
RUN pip install rouge
RUN pip install networkx
RUN pip install types-tensorflow
RUN pip install tensorflow
RUN pip install tensorflow-addons
RUN pip install autopep8

RUN useradd -c '' -m -d /home/u -G sudo -s /bin/bash u
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER u
WORKDIR /home/u

CMD ["/bin/bash"]
