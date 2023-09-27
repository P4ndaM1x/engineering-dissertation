FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /content

COPY requirements.txt requirements.txt

RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8888