FROM python:3.12

WORKDIR /home/langchain_tutorial
COPY requirements.txt .
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip install -r requirements.txt

COPY tutorial.py Data_Engineer_CV_Manuel_Juarez.pdf /home/langchain_tutorial


CMD ["tail","-f","/dev/null"]

