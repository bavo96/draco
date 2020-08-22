FROM python:3.6

# INSTALL DEPENDENCIES
RUN apt-get update
RUN apt-get -y install \
    git \
    vim \
    telnet

RUN git clone https://github.com/bavo96/draco.git

RUN chmod +x draco/download_model.sh
RUN ./draco/download_model.sh

RUN pip3 install --upgrade pip
RUN pip3 install -r draco/requirements.txt

CMD [ "python draco/predict/full_pipeline.py" ]
#ENTRYPOINT [ "python draco/predict/full_pipeline.py" ]