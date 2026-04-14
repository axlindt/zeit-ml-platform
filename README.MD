# ZEIT ML Platform

Lokale MLOps-Plattform als Lösung der ZEIT ML/AI Platform Bewerbungsaufgabe.
```
┌─────────────────────────────────────┐
│           k3d Cluster               │
│                                     │
│  ┌─────────┐      ┌──────────────┐  │
│  │  MinIO  │◄─────│    MLflow    │  │
│  │(Storage)│      │  (Tracking)  │  │
│  └─────────┘      └──────────────┘  │
│                          ▲          │
│                          │          │
│              ┌───────────────────┐  │
│              │  Churn Model Job  │  │
│              │ (train_and_score) │  │
│              └───────────────────┘  │
└─────────────────────────────────────┘
```

## Voraussetzungen

- Docker Desktop
- kubectl, helm, k3d, terraform (macOS: `brew install terraform kubectl helm k3d`)

## Schritt 1: Kubernetes Cluster aufsetzen

Ich nutze k3d für einen lokalen Kubernetes Cluster in Docker. Das vermeidet GCP Billing-Setup und funktioniert für dieses Beispiel genauso wie GKE. In Production würde das Cluster selbst via Terraform als `google_container_cluster` erstellt werden.

```bash
k3d cluster create zeit-ml-platform
kubectl get nodes
```

## Schritt 2: Storage bereitstellen

Für den Storage nutze ich MinIO, einen lokalen Object Store der das S3-Protokoll spricht und damit als GCS-Ersatz funktioniert. Alle Plattform-Komponenten (MinIO, MLflow) laufen im Namespace `mlflow`.

Terraform erstellt zwei Buckets: `mlflow-artifacts` für MLflow Artefakte und `ml-data` für Trainings- und Scoring-Daten.

Da der MinIO Terraform Provider eine direkte Verbindung braucht, muss MinIO erst laufen bevor die Buckets erstellt werden können:

```bash
cd infra/terraform
terraform init
terraform apply  # deployt MinIO

kubectl port-forward svc/minio 9000:9000 -n mlflow &
terraform apply  # erstellt Buckets und lädt Daten hoch
```

## Schritt 3: MLflow deployen
 
MLflow ist der Tracking Server für Experimente, Metriken und Modell-Artefakte. Ich deploye ihn via einem eigenen Helm Chart (einem Kubernetes Deployment-Paket) statt einem fertigen Chart wie Bitnami, weil der Bitnami Chart seine eigene Datenbank und seinen eigenen Storage mitbringt was mit unserem MinIO kollidiert.
 
MLflow wird so konfiguriert dass er Artefakte in dem von Terraform erstellten Bucket `mlflow-artifacts` speichert. Mit `--serve-artifacts` fungiert MLflow als Proxy zwischen Browser und MinIO, so dass der Browser nie direkt auf MinIO zugreifen muss.
 
Da das offizielle MLflow Image kein `boto3` hat (das S3 SDK das MLflow braucht um mit MinIO zu reden), bauen wir ein eigenes Image:
 
```bash
docker build -t zeit-mlflow -f infra/docker/Dockerfile.mlflow .
k3d image import zeit-mlflow -c zeit-ml-platform
```
 
Dann MLflow via Terraform deployen:
 
```bash
cd infra/terraform
terraform apply
 
# MLflow UI erreichbar unter
kubectl port-forward svc/mlflow 5001:5000 -n mlflow
```

## Schritt 4: Modell trainieren
 
Das Training Script wird als Docker Container gebaut und als Kubernetes Job ausgeführt. Es liest die Trainingsdaten direkt aus MinIO und loggt Parameter, Metriken und das Modell-Artefakt in MLflow.
 
```bash
docker build -t zeit-churn-model .
k3d image import zeit-churn-model -c zeit-ml-platform
 
kubectl apply -f infra/k8s/training-job.yaml
kubectl logs -n mlflow job/churn-training -f
```
 
## Schritt 5: Scoring
 
Der Scoring Job lädt automatisch den letzten Training Run aus MLflow. Optional kann eine spezifische Run ID via `MODEL_RUN_ID` angegeben werden (in `scoring-job.yaml` auskommentiert).
 
```bash
kubectl apply -f infra/k8s/scoring-job.yaml
kubectl logs -n mlflow job/churn-scoring -f
```
 
Die berechneten Scores sind dann in zwei Formen verfügbar:
- Als CSV in MinIO (`s3://ml-data/churn-xgbrf-{model_id}_output.csv`)
- Als downloadbares Artefakt im MLflow Scoring Run

## Teardown
 
```bash
kubectl delete -f infra/k8s/
cd infra/terraform && terraform destroy
k3d cluster delete zeit-ml-platform
```

## Limitierungen und Ausblick 
 
Dieses Setup ist als Proof of Concept gedacht. Ein paar Gedanken dazu was sich ändern wuerde wenn das in Production geht und das Team wächst:
 
**GCP Shift**
Die `/infra` Struktur bleibt ähnlich, hauptsaechlich Provider und Endpoints aendern sich: k3d wird zu GKE, MinIO zu GCS, SQLite zu Cloud SQL. Docker Images wuerden in der GCP Artifact Registry gespeichert statt lokal importiert.
 
**Abstraktion fuer Data Scientists**
Im Moment muessen Nutzer:innen Kubernetes YAML Files kennen. Sinnvoll wäre eine klarere Trennung in zwei Repos:
 
`zeit-ml-platform` (dieses Repo, Platform Team) -- Terraform, Helm Charts, Base Images, CLI Tool
 
`zeit-ml-project-template` (Data Scientists clonen das) -- eigener Trainingscode, Dockerfile das auf der Base Image aufbaut, pyproject.toml
 
Das CLI verbindet beide Welten, es kennt die Projektstruktur von `zeit-ml-project-template`, das Cluster und die Konventionen. Data Scientists müssen weder Kubernetes noch Terraform kennen sondern verwenden das CLI tool, zum Beispiel:
 
```bash
zeit-ml new churn-model    # neues Projekt aus Template aufsetzen
zeit-ml train              # baut Image, startet Training Job, streamt Logs
zeit-ml score              # startet Scoring Job
zeit-ml debug              # interaktive Research Pod Session
```
 
Darunter laufen verschiedene Pod-Typen mit derselben Base Image aber unterschiedlichen Ressourcen: ein leichter Research Pod fuer Exploration, ein GPU Training Pod und ein CPU-only Scoring Pod. Weil alle Data Scientists das CLI nutzen sind alle Projekte gleich strukturiert, es gibt keine individuellen Kubernetes Konfigurationen mehr.
 
**Credentials**
Optimal wäre Workload Identity statt Secrets. Die Idee: Nutzer:innen loggen sich lokal via `gcloud auth login` ein, das gibt ihnen GCP IAM Rechte, z.B. "darf Jobs im Namespace ml-jobs starten". Wenn sie dann `kubectl apply` (bzw. die o.g. CLI Befehle) ausfuehren, startet der Pod mit einem vordefinierten GCP Service Account der nur minimale Rechte hat z.B. "darf nur aus ml-data lesen, nicht schreiben". Nutzer:innen können diesen Service Account nicht selbst ändern, dieser wird vom Admin vordefiniert. So gibt es zwei unabhängige Sicherheitsebenen: was die Nutzer:innen tun dürfen, und was der Pod tun darf. 
 
**Ressourcen und Billing**
Kubernetes Resource Quotas pro Namespace setzen harte Limits: Wie viele Pods gleichzeitig laufen duerfen, wie viel CPU, RAM und GPU ein Namespace verbrauchen darf. Ein neuer Job der diese Limits ueberschreiten würde wird  abgelehnt. Das verhindert dass ein einzelner Training Run den ganzen Cluster lahmlegt.
 
Fuer Billing reicht oft ein einziges GCP Projekt mit Labels auf allen Ressourcen (`team: data-science`, `project: churn-model`, `env: production`). GCP Billing kann dann nach Labels filtern, man sieht genau was welches Team oder Projekt kostet ohne mehrere GCP Projekte verwalten zu müssen. Separate GCP Projekte statt Labels machen Sinn wenn man noch stärkere Isolation möchte.

GCP Budget Alerts schicken Emails wenn 50%, 90% oder 100% eines gesetzten Budgets erriehct sind, stoppen aber nichts automatisch wie die  Kubernetes Quotas.
 