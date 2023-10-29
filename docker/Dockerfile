FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /content

COPY requirements.txt requirements.txt

RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
# RUN sed -i '6 i ,"token": "1b0a9bf18768c6224aa65b7379984803784eeb2c06fc7e99"' /root/.jupyter/jupyter_notebook_config.json
RUN mkdir /root/.jupyter
RUN printf "c = get_config()\nc.ServerApp.token = \"1b0a9bf18768c6224aa65b7379984803784eeb2c06fc7e99\"\n" > /root/.jupyter/jupyter_server_config.py

EXPOSE 8888