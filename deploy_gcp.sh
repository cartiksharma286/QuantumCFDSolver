#!/bin/bash
set -e

# Configuration
PROJECT_ID=$(gcloud config get-value project)
IMAGE_NAME="quantum-cfd-solver"
REGION="us-central1"
CLUSTER="quantum-cluster"

echo "Deploying to Project: $PROJECT_ID"

# 1. Build Docker Image
echo "Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME:latest -f QuantumCFD/Dockerfile .

# 2. Push to Container Registry
echo "Pushing image to GCR..."
gcloud auth configure-docker
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME:latest

# 3. Update Manifest (Simple sed replacement for Project ID)
echo "Updating Kubernetes Manifest..."
sed -i "s|PROJECT_ID|$PROJECT_ID|g" QuantumCFD/deployment/k8s-deployment.yaml

# 4. Deploy to GKE
echo "Deploying to Kubernetes..."
# Assuming user is already connected to the cluster, if not:
# gcloud container clusters get-credentials $CLUSTER --region $REGION
kubectl apply -f QuantumCFD/deployment/k8s-deployment.yaml

echo "Deployment initiated."
kubectl get pods -l app=quantum-cfd
