### Train an XGBoost Random Forest on the training data ###
#
# Unterstuetzt zwei Modi via MODE Umgebungsvariable:
#   MODE=train  -- trainiert das Modell und loggt alles in MLflow
#   MODE=score  -- lädt das letzte Modell aus MLflow und scored neue Daten
#
# Umgebungsvariablen:
#   MODE                  -- "train" oder "score" (default: "train")
#   TRAIN_DATA_PATH       -- Pfad zur Trainings-CSV (s3:// oder lokal) [required fuer train]
#   SCORE_DATA_PATH       -- Pfad zur Scoring-CSV (s3:// oder lokal) [required fuer score]
#   OUTPUT_PATH           -- Pfad fuer Scoring-Ergebnisse [required fuer score]
#   AWS_ACCESS_KEY_ID     -- MinIO Access Key [required]
#   AWS_SECRET_ACCESS_KEY -- MinIO Secret Key [required]
#   MLFLOW_TRACKING_URI   -- MLflow Server URL (default: http://mlflow:5000)
#   MLFLOW_S3_ENDPOINT_URL -- MinIO Endpoint (default: http://minio:9000)
#   MODEL_NAME            -- Name des Modell-Checkpoints (default: churn-xgbrf)
#   MODEL_RUN_ID          -- MLflow Run ID fuer Scoring (optional, default: letzter Run)
#   VAL_SIZE              -- Anteil Validierungsdaten (default: 0.2)

import os
import platform
import joblib
import tempfile
import pandas as pd
import mlflow
import s3fs
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRFClassifier


def require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Required environment variable {key} is not set")
    return value


# Required for both modes
AWS_KEY = require_env("AWS_ACCESS_KEY_ID")
AWS_SECRET = require_env("AWS_SECRET_ACCESS_KEY")

# Optional
MODE = os.getenv("MODE", "train")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
S3_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
MODEL_NAME = os.getenv("MODEL_NAME", "churn-xgbrf")
VAL_SIZE = float(os.getenv("VAL_SIZE", "0.2"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

features = ["objekt_name", "aboform_name", "lesedauer", "zahlung_weg_name"]
target = "churn"


def get_s3fs():
    return s3fs.S3FileSystem(
        key=AWS_KEY,
        secret=AWS_SECRET,
        client_kwargs={"endpoint_url": S3_ENDPOINT},
    )


def read_csv(path):
    if path.startswith("s3://"):
        fs = get_s3fs()
        with fs.open(path, "r") as f:
            return pd.read_csv(f, sep=';')
    return pd.read_csv(path, sep=';')


def log_system_info():
    mlflow.set_tag("python_version", platform.python_version())
    mlflow.set_tag("platform", platform.platform())
    mlflow.set_tag("model_name", MODEL_NAME)
    mlflow.set_tag("mlflow_tracking_uri", MLFLOW_TRACKING_URI)


def log_dataset_metrics(prefix, y):
    """Log class distribution for a dataset split."""
    mlflow.log_metric(f"{prefix}/n_samples", len(y))
    mlflow.log_metric(f"{prefix}/n_positive", int(y.sum()))
    mlflow.log_metric(f"{prefix}/n_negative", int((y == 0).sum()))
    mlflow.log_metric(f"{prefix}/positive_rate", float(y.mean()))


def log_classification_metrics(prefix, y_true, y_pred, y_prob):
    """Log full classification metrics for a dataset split."""
    mlflow.log_metric(f"{prefix}/accuracy", float(accuracy_score(y_true, y_pred)))
    mlflow.log_metric(f"{prefix}/f1", float(f1_score(y_true, y_pred)))
    mlflow.log_metric(f"{prefix}/auc", float(roc_auc_score(y_true, y_prob)))
    mlflow.log_metric(f"{prefix}/precision_churner", float(precision_score(y_true, y_pred)))
    mlflow.log_metric(f"{prefix}/recall_churner", float(recall_score(y_true, y_pred)))
    mlflow.log_metric(f"{prefix}/precision_non_churner", float(precision_score(y_true, y_pred, pos_label=0)))
    mlflow.log_metric(f"{prefix}/recall_non_churner", float(recall_score(y_true, y_pred, pos_label=0)))


def train():
    train_data_path = require_env("TRAIN_DATA_PATH")

    mlflow.set_experiment("churn-training")
    mlflow.xgboost.autolog(log_models=False)

    print(f"Lade Trainingsdaten von {train_data_path}")
    df = read_csv(train_data_path)

    # Identify categorical features for one-hot encoding
    categorical_features = [col for col in features if df[col].dtype == 'object']

    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    X = df_encoded.drop(columns=target)
    y = df_encoded[target]

    # Train/Validation Split — stratified um Klassenbalance zu erhalten
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=42, stratify=y
    )

    # Class balancing via scale_pos_weight
    n_negative = int((y_train == 0).sum())
    n_positive = int(y_train.sum())
    scale_pos_weight = n_negative / n_positive

    params = {
        "n_estimators": 100,
        "max_depth": 4,
        "random_state": 42,
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight,
    }

    with mlflow.start_run(run_name=MODEL_NAME) as run:
        run_id = run.info.run_id
        model_filename = f"{MODEL_NAME}-{run_id}.joblib"

        log_system_info()
        mlflow.set_tag("train_data_path", train_data_path)
        mlflow.set_tag("model_filename", model_filename)

        # model/ params
        mlflow.log_params({f"model/{k}": v for k, v in params.items()})
        mlflow.log_param("model/val_size", VAL_SIZE)
        mlflow.log_param("model/n_features", len(X.columns))
        mlflow.log_param("model/categorical_features", str(categorical_features))

        # dataset/ metrics
        log_dataset_metrics("dataset/train", y_train)
        log_dataset_metrics("dataset/val", y_val)
        mlflow.log_metric("dataset/scale_pos_weight", scale_pos_weight)

        model = XGBRFClassifier(**params, use_label_encoder=False)
        model.fit(X_train, y_train)

        # train_metrics/
        train_pred = model.predict(X_train)
        train_prob = model.predict_proba(X_train)[:, 1]
        log_classification_metrics("train_metrics", y_train, train_pred, train_prob)

        # val_metrics/
        val_pred = model.predict(X_val)
        val_prob = model.predict_proba(X_val)[:, 1]
        log_classification_metrics("val_metrics", y_val, val_pred, val_prob)

        print(f"Train Accuracy: {accuracy_score(y_train, train_pred):.4f} | Val Accuracy: {accuracy_score(y_val, val_pred):.4f}")
        print(f"Train F1:       {f1_score(y_train, train_pred):.4f} | Val F1:       {f1_score(y_val, val_pred):.4f}")
        print(f"Train AUC:      {roc_auc_score(y_train, train_prob):.4f} | Val AUC:      {roc_auc_score(y_val, val_prob):.4f}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Validation predictions mit Run ID
            val_preds_path = os.path.join(tmpdir, f"val_predictions-{run_id}.csv")
            y_val.to_frame().assign(
                predicted=val_pred,
                probability=val_prob
            ).to_csv(val_preds_path, index=False)
            mlflow.log_artifact(val_preds_path, artifact_path="predictions")

            # Model checkpoint mit Run ID
            model_path = os.path.join(tmpdir, model_filename)
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="model")

        print(f"Run geloggt in MLflow: {MLFLOW_TRACKING_URI}")


def score():
    score_data_path = require_env("SCORE_DATA_PATH")
    output_path = require_env("OUTPUT_PATH")

    client = mlflow.tracking.MlflowClient()

    # MODEL_RUN_ID optional -- default ist der letzte Training Run
    model_run_id = os.getenv("MODEL_RUN_ID")
    if not model_run_id:
        experiment = client.get_experiment_by_name("churn-training")
        runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
        if not runs:
            raise ValueError("Kein Training Run gefunden -- erst trainieren!")
        model_run_id = runs[0].info.run_id
        print(f"Kein MODEL_RUN_ID angegeben -- nutze letzten Run: {model_run_id}")

    run = client.get_run(model_run_id)
    model_name = run.data.tags.get("model_name", MODEL_NAME)
    model_filename = run.data.tags.get("model_filename", f"{model_name}.joblib")

    mlflow.set_experiment("churn-scoring")

    print(f"Lade Scoring-Daten von {score_data_path}")
    score_df = read_csv(score_data_path)

    print(f"Lade Modell '{model_filename}' aus Run {model_run_id}")

    with mlflow.start_run(run_name=f"{model_name}-scoring") as scoring_run:
        log_system_info()
        mlflow.set_tag("training_run_id", model_run_id)
        mlflow.set_tag("model_name_used", model_name)
        mlflow.set_tag("model_filename_used", model_filename)
        mlflow.set_tag("score_data_path", score_data_path)
        mlflow.set_tag("model_artifact_uri", f"{run.info.artifact_uri}/model/{model_filename}")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = client.download_artifacts(model_run_id, f"model/{model_filename}", tmpdir)
            model = joblib.load(model_path)

        categorical_features = [col for col in features if score_df[col].dtype == 'object']
        score_df_encoded = pd.get_dummies(score_df, columns=categorical_features, drop_first=True)

        # Align columns - this is crucial to ensure that the scoring data has the exact same columns as the training data
        # This handles cases where a category might be present in train but not in score, or vice-versa.
        expected_columns = list(model.get_booster().feature_names)
        missing_cols_in_score = set(expected_columns) - set(score_df_encoded.columns)
        for c in missing_cols_in_score:
            score_df_encoded[c] = 0

        # Ensure the order of columns is the same as in X_train
        X_score = score_df_encoded[expected_columns]

        # Compute churn probabilities
        probabilities = model.predict_proba(X_score)[:, 1]
        score_df["churn_probability"] = probabilities

        probs = score_df["churn_probability"]
        mlflow.log_metric("scores/n_records", len(score_df))
        mlflow.log_metric("scores/mean_probability", float(probs.mean()))
        mlflow.log_metric("scores/median_probability", float(probs.median()))
        mlflow.log_metric("scores/std_probability", float(probs.std()))
        mlflow.log_metric("scores/pct_high_risk", float((probs > 0.5).mean()))
        mlflow.log_metric("scores/pct_medium_risk", float(((probs >= 0.3) & (probs < 0.5)).mean()))
        mlflow.log_metric("scores/pct_low_risk", float((probs < 0.3).mean()))

        # Scores werden via MLflow als Artefakt gespeichert -- kein direkter MinIO Zugriff noetig
        with tempfile.TemporaryDirectory() as tmpdir:
            scores_filename = f"{model_name}-{model_run_id}_scores.csv"
            local_output = os.path.join(tmpdir, scores_filename)
            score_df.to_csv(local_output, index=False)
            mlflow.log_artifact(local_output, artifact_path="scores")

        print(score_df[['id'] + features + ["churn_probability"]].head(10))
        print(f"\n{len(score_df)} Records gescored -> gespeichert in {output_path}")
        print(f"Scores auch in MLflow Run {scoring_run.info.run_id} verfuegbar")


if __name__ == "__main__":
    if MODE == "train":
        train()
    elif MODE == "score":
        score()
    else:
        raise ValueError(f"Unbekannter MODE: {MODE}. Nutze 'train' oder 'score'.")
