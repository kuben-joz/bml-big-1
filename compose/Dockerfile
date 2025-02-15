FROM python:latest

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt requirements.txt
RUN apt-get update && \
  apt-get install -y openssh-server openmpi-bin openmpi-common libopenmpi-dev build-essential && \
  mkdir /var/run/sshd

RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config
RUN echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config

COPY key /root/.ssh/id_rsa
COPY key.pub /root/.ssh/authorized_keys
RUN chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
RUN echo "Host *" >> /root/.ssh/config
RUN echo "  IdentityFile /root/.ssh/key"
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY setup.sh setup.sh
RUN chmod +x setup.sh

CMD ["/usr/sbin/sshd", "-D"]