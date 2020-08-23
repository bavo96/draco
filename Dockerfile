FROM python:3.6.12-slim

# INSTALL DEPENDENCIES
RUN apt-get update
RUN apt-get -y install \
    git \
    vim \
    telnet \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0

WORKDIR /home/

RUN git clone https://github.com/bavo96/draco.git /home/draco
RUN ls /home/draco
RUN chmod +x /home/draco/download_model.sh
RUN /home/draco/download_model.sh

RUN pip3 install --upgrade pip
RUN pip3 install -r /home/draco/requirements-cpu.txt

CMD [ "python3", "draco/predict/full_pipeline.py" ]