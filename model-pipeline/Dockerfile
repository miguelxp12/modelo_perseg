FROM python:3.10

ENV PYTHONUNBUFFERED=TRUE

RUN apt-get update && apt-get install -y wget &&     wget https://bootstrap.pypa.io/get-pip.py &&     python3.10 get-pip.py &&     rm get-pip.py &&     apt-get remove -y wget &&     apt-get clean

COPY src .
COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3","run_step.py"]
CMD []
