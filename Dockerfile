FROM python:3.11-slim

WORKDIR /app

# Dependencies vor dem Script installieren damit Docker Layer Caching greift
COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY train_and_score.py .

ENV MODE=train
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV TRAIN_DATA_PATH=s3://ml-data/train_data.csv
ENV SCORE_DATA_PATH=s3://ml-data/scoring_data.csv
ENV OUTPUT_PATH=s3://ml-data/scored_output.csv

CMD ["python", "train_and_score.py"]
