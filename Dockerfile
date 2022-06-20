FROM tensorflow/tensorflow:2.4.3-gpu
RUN apt install git
RUN cd /
RUN git clone -b note https://github.com/stu00608/LoFi-Loop-Generator.git
RUN cd LoFi-Loop-Generator
RUN pip install -r requirements.txt
CMD [ "bash" ]