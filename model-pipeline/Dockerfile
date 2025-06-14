FROM python:3.10

ENV PYTHONUNBUFFERED=TRUE
# Define a working directory
WORKDIR /app

RUN apt-get update && apt-get install -y wget &&     wget https://bootstrap.pypa.io/get-pip.py &&     python3.10 get-pip.py &&     rm get-pip.py &&     apt-get remove -y wget &&     apt-get clean

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
