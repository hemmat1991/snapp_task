FROM python:3.8-slim-buster
RUN apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Tehran /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update && apt-get -y install cron nano htop


WORKDIR /usr/src/app
COPY . /usr/src/app/

COPY requirements.txt /usr/src/app/
RUN pip install --index-url https://repos.myket.ir/repository/pypi-registry/simple -r requirements.txt
CMD ["bash", "exec.sh"]

