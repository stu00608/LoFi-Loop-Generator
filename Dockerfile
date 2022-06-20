FROM tensorflow/tensorflow:2.4.0-gpu

ADD . /lofi

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt update -y
RUN pip install --upgrade pip

WORKDIR /lofi
RUN mkdir -p outputs
RUN mkdir -p models
RUN mkdir -p data

RUN pip install -r requirements.txt
ENTRYPOINT [ "/bin/bash" ]