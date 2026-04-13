terraform {
  required_providers {
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    minio = {
      source  = "aminueza/minio"
      version = "~> 2.0"
    }
  }
}

provider "kubernetes" {
  config_path    = "~/.kube/config"
  config_context = "k3d-zeit-ml-platform"
}

provider "helm" {
  kubernetes {
    config_path    = "~/.kube/config"
    config_context = "k3d-zeit-ml-platform"
  }
}

# MinIO läuft im Cluster — Terraform verbindet sich via port-forward.
# Bevor terraform apply: kubectl port-forward svc/minio 9000:9000 -n mlflow
provider "minio" {
  minio_server   = "localhost:9000"
  minio_user     = var.minio_access_key
  minio_password = var.minio_secret_key
  minio_ssl      = false
}
