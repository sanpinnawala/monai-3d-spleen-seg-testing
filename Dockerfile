FROM python:3.9-slim-buster
WORKDIR /.
COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt
RUN pip install --upgrade pip
RUN apt-get update \
&& apt-get install -y sudo
RUN pip3 install -r requirements.txt
RUN pip3 install -r requirements-dev.txt
COPY . .
CMD ["python3", "main.py"]