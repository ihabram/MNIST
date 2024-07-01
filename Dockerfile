FROM python:3.8

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./mnist_api /code/mnist_api

EXPOSE 8000

CMD ["uvicorn", "mnist_api.server:app", "--host", "0.0.0.0", "--port", "8000"]