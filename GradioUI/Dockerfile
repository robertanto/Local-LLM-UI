FROM python:3

WORKDIR /home 

ADD requirements.txt /home
RUN pip install --no-cache-dir -r requirements.txt

COPY src /home

CMD ["python3","main.py"]