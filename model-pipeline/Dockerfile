FROM --platform=linux/amd64 python:3.10-slim-bullseye

RUN apt-get update && apt-get install -y wget && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py && \
    apt-get remove -y wget && \
    apt-get clean
RUN apt-get update && apt-get install -y libgomp1

# Copy the environment file first
COPY forecastperseg.env .

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src .

ENTRYPOINT ["python3","run_step.py"]
CMD []
