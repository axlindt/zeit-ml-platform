# ZEIT ML Platform

Lokale MLOps-Plattform als Lösung der ZEIT ML/AI Platform Bewerbungsaufgabe.

## Voraussetzungen

- Docker Desktop
- kubectl, helm, k3d, terraform (macOS: `brew install terraform kubectl helm k3d`)

## Schritt 1: Kubernetes Cluster aufsetzen

Wir nutzen k3d für einen lokalen Kubernetes Cluster in Docker, kostenlos und identisch zu GKE in der Verwedung. In Production würde das Cluster selbst via Terraform als `google_container_cluster` erstellt werden.

```bash
k3d cluster create zeit-ml-platform
kubectl get nodes
```
