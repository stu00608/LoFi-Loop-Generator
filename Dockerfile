FROM tensorflow/tensorflow:2.4.0-gpu
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

ARG wandbapi
ARG expname
ARG data_size
ARG batch_size
ARG epochs

RUN apt update -y
RUN pip install --upgrade pip

RUN apt install -y git
WORKDIR /
RUN git clone -b note https://github.com/stu00608/LoFi-Loop-Generator.git
WORKDIR /LoFi-Loop-Generator
RUN mkdir outputs
RUN mkdir models
RUN mkdir data
RUN git submodule init
RUN git submodule update

RUN pip install -r requirements.txt
ENTRYPOINT [ "entry.sh" ]