FROM nvcr.io/nvidia/pytorch:23.06-py3 as base
ENV DEBIAN_FRONTEND=noninteractive

RUN << EOF
apt update -y
apt install -y vim ranger build-essential wget curl
mkdir /root/LightningTS
EOF

COPY . /root/LightningTS
WORKDIR /root/LightningTS
RUN pip3 install -r requirements.txt
