FROM python:3.8-slim-buster

ARG DATASET_TYPE
ARG MODEL_TYPE
ARG MODEL_NAME

WORKDIR /app

COPY ../requirements.txt /app/
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY meta_learn/pretrained_meta_regressors/${DATASET_TYPE}/${MODEL_TYPE}/${MODEL_NAME}.joblib /app/
COPY ./service /app/

CMD ["python", "./service/main.py", "-path2model", ${MODEL_NAME}.joblib]