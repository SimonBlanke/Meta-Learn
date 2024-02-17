FROM python:3.8-slim-buster
ARG path2meta_regressor

WORKDIR /app

COPY ../requirements.txt /app/
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY path2meta_regressor /app/
COPY ./service /app/

CMD ["python", "./service/main.py"]