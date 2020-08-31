FROM python:3.6.12-slim

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN rm -rf /var/lib/apt/lists/*

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

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda install python=3.6 -y
RUN conda install faiss-cpu -c pytorch -y

WORKDIR /home/

RUN git clone https://github.com/bavo96/draco.git /home/draco
RUN chmod +x /home/draco/download_model.sh

# INSTALL PYTHON PACKAGES
RUN pip3 install --upgrade pip
RUN pip3 install -r /home/draco/requirements-cpu.txt

#DOWNLOAD MODELS
RUN /home/draco/download_model.sh

#CMD [ "python3", "draco/predict/full_pipeline.py" ]