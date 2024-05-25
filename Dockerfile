FROM python:3.12

WORKDIR /home/langchain_tutorial
COPY requirements.txt .
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip install -r requirements.txt

COPY  papers_resistencia_interfaz /home/langchain_tutorial/papers_resistencia_interfaz
COPY tutorial.py /home/langchain_tutorial

CMD ["tail","-f","/dev/null"]

