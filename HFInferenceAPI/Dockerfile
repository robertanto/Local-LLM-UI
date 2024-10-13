FROM python:3

WORKDIR /home 

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && export PATH="$HOME/.cargo/bin:$PATH"

ADD requirements.txt /home
RUN pip install --no-cache-dir -r requirements.txt

COPY src /home

CMD ["uvicorn","main:app","--reload", "--proxy-headers", "--host", "0.0.0.0", "--port", "8080"]