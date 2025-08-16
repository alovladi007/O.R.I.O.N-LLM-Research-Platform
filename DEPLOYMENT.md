# ORION Platform Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Database Migration](#database-migration)
6. [SSL/TLS Configuration](#ssltls-configuration)
7. [Monitoring Setup](#monitoring-setup)
8. [Backup and Recovery](#backup-and-recovery)
9. [Scaling Guidelines](#scaling-guidelines)
10. [Troubleshooting](#troubleshooting)

## Overview

The ORION platform is a cloud-native, microservices-based application designed for high availability and scalability. This guide covers deployment to Kubernetes clusters in production environments.

### Architecture Components

- **API Service**: FastAPI backend with async support
- **Frontend**: Next.js SSR application
- **Worker Service**: Celery workers for background tasks
- **Databases**: PostgreSQL (primary), Redis (cache), Neo4j (graph), Elasticsearch (search)
- **Storage**: MinIO for object storage
- **Monitoring**: Prometheus, Grafana, Jaeger
- **Ingress**: NGINX with SSL termination

## Prerequisites

### Required Tools

```bash
# Kubernetes CLI
kubectl version --client
# Output: Client Version: v1.28.0

# Helm package manager
helm version
# Output: v3.13.0

# Docker
docker --version
# Output: Docker version 24.0.0

# Terraform (optional, for infrastructure provisioning)
terraform --version
# Output: Terraform v1.6.0
```

### Cloud Provider Setup

#### AWS EKS

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create cluster
eksctl create cluster \
  --name orion-production \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed
```

#### Google GKE

```bash
# Create cluster
gcloud container clusters create orion-production \
  --zone us-central1-a \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10 \
  --machine-type n2-standard-4 \
  --enable-autorepair \
  --enable-autoupgrade
```

#### Azure AKS

```bash
# Create resource group
az group create --name orion-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group orion-rg \
  --name orion-production \
  --node-count 3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10 \
  --generate-ssh-keys
```

## Infrastructure Setup

### 1. Install Required Operators

```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Install Cert-Manager for SSL
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Install Elasticsearch Operator
kubectl create -f https://download.elastic.co/downloads/eck/2.9.0/crds.yaml
kubectl apply -f https://download.elastic.co/downloads/eck/2.9.0/operator.yaml

# Install PostgreSQL Operator
kubectl apply -f https://raw.githubusercontent.com/cloudnative-pg/cloudnative-pg/release-1.20/releases/cnpg-1.20.0.yaml
```

### 2. Create Namespaces and Secrets

```bash
# Create namespace
kubectl create namespace orion-platform

# Create secrets
kubectl create secret generic orion-secrets \
  --from-literal=database-url='postgresql+asyncpg://orion:secure_password@orion-postgres:5432/orion_db' \
  --from-literal=redis-url='redis://:redis_password@orion-redis:6379/0' \
  --from-literal=neo4j-uri='bolt://orion-neo4j:7687' \
  --from-literal=neo4j-password='neo4j_secure_password' \
  --from-literal=secret-key='your-secret-key-here' \
  --from-literal=postgres-user='orion' \
  --from-literal=postgres-password='postgres_secure_password' \
  --from-literal=postgres-replication-password='replication_password' \
  --from-literal=minio-access-key='minio_access_key' \
  --from-literal=minio-secret-key='minio_secret_key' \
  --from-literal=openai-api-key='your-openai-api-key' \
  -n orion-platform

# Create image pull secret (if using private registry)
kubectl create secret docker-registry regcred \
  --docker-server=ghcr.io \
  --docker-username=your-github-username \
  --docker-password=your-github-token \
  --docker-email=your-email@example.com \
  -n orion-platform
```

### 3. Configure Storage Classes

```yaml
# storage-class.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs  # or appropriate for your cloud
parameters:
  type: gp3
  iops: "16000"
  throughput: "1000"
reclaimPolicy: Retain
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

```bash
kubectl apply -f storage-class.yaml
```

## Kubernetes Deployment

### 1. Deploy Infrastructure Components

```bash
# Deploy databases
kubectl apply -f k8s/base/postgres-deployment.yaml
kubectl apply -f k8s/base/redis-deployment.yaml
kubectl apply -f k8s/base/neo4j-deployment.yaml
kubectl apply -f k8s/base/elasticsearch-deployment.yaml
kubectl apply -f k8s/base/minio-deployment.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l component=postgres -n orion-platform --timeout=300s
```

### 2. Deploy Application Components

```bash
# Deploy API service
kubectl apply -f k8s/base/api-deployment.yaml

# Deploy frontend
kubectl apply -f k8s/base/frontend-deployment.yaml

# Deploy workers
kubectl apply -f k8s/base/worker-deployment.yaml

# Deploy ingress
kubectl apply -f k8s/base/ingress.yaml
```

### 3. Verify Deployment

```bash
# Check pod status
kubectl get pods -n orion-platform

# Check services
kubectl get svc -n orion-platform

# Check ingress
kubectl get ingress -n orion-platform

# View logs
kubectl logs -f deployment/orion-api -n orion-platform
```

## Database Migration

### 1. Run Initial Migrations

```bash
# Execute migrations
kubectl exec -it deployment/orion-api -n orion-platform -- \
  alembic upgrade head

# Seed initial data
kubectl exec -it deployment/orion-api -n orion-platform -- \
  python scripts/seed_data.py
```

### 2. Create Database Backups

```bash
# PostgreSQL backup
kubectl exec -it orion-postgres-0 -n orion-platform -- \
  pg_dump -U orion orion_db > backup-$(date +%Y%m%d-%H%M%S).sql

# Neo4j backup
kubectl exec -it orion-neo4j-0 -n orion-platform -- \
  neo4j-admin backup --database=neo4j --to=/backups/neo4j-$(date +%Y%m%d-%H%M%S)
```

## SSL/TLS Configuration

### 1. Configure Let's Encrypt

```yaml
# letsencrypt-issuer.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@orion-platform.ai
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

```bash
kubectl apply -f letsencrypt-issuer.yaml
```

### 2. Verify SSL Certificate

```bash
# Check certificate status
kubectl describe certificate orion-tls -n orion-platform

# Test HTTPS endpoint
curl -I https://orion-platform.ai
```

## Monitoring Setup

### 1. Configure Prometheus Scraping

```yaml
# prometheus-servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: orion-api
  namespace: orion-platform
spec:
  selector:
    matchLabels:
      app: orion
      component: api
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### 2. Import Grafana Dashboards

```bash
# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Login with admin/prom-operator
# Import dashboards from k8s/monitoring/dashboards/
```

### 3. Set Up Alerts

```yaml
# alerts.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: orion-alerts
  namespace: orion-platform
spec:
  groups:
  - name: orion.rules
    interval: 30s
    rules:
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: High error rate detected
    - alert: PodCrashLooping
      expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: Pod is crash looping
```

## Backup and Recovery

### 1. Automated Backups

```yaml
# backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
  namespace: orion-platform
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: orion-backup:latest
            command:
            - /bin/bash
            - -c
            - |
              # Backup PostgreSQL
              pg_dump -h orion-postgres -U orion orion_db | gzip > /backups/postgres-$(date +\%Y\%m\%d).sql.gz
              # Upload to S3
              aws s3 cp /backups/postgres-$(date +\%Y\%m\%d).sql.gz s3://orion-backups/postgres/
          restartPolicy: OnFailure
```

### 2. Disaster Recovery

```bash
# Restore PostgreSQL from backup
kubectl exec -i orion-postgres-0 -n orion-platform -- \
  psql -U orion orion_db < backup-20240115-120000.sql

# Restore Neo4j
kubectl exec -it orion-neo4j-0 -n orion-platform -- \
  neo4j-admin restore --from=/backups/neo4j-20240115-120000 --database=neo4j
```

## Scaling Guidelines

### 1. Horizontal Pod Autoscaling

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: orion-api-hpa
  namespace: orion-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: orion-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 2. Cluster Autoscaling

```bash
# Enable cluster autoscaler
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml

# Configure autoscaler
kubectl -n kube-system annotate deployment.apps/cluster-autoscaler \
  cluster-autoscaler.kubernetes.io/safe-to-evict="false"
```

### 3. Database Scaling

```bash
# Scale PostgreSQL replicas
kubectl scale statefulset orion-postgres --replicas=5 -n orion-platform

# Add read replicas
kubectl patch statefulset orion-postgres -n orion-platform --type='json' \
  -p='[{"op": "add", "path": "/spec/template/spec/containers/0/env/-", "value": {"name": "POSTGRES_REPLICATION_MODE", "value": "slave"}}]'
```

## Troubleshooting

### Common Issues

#### 1. Pod Crash Loops

```bash
# Check pod logs
kubectl logs -f pod-name -n orion-platform --previous

# Describe pod for events
kubectl describe pod pod-name -n orion-platform

# Check resource limits
kubectl top pods -n orion-platform
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
kubectl exec -it deployment/orion-api -n orion-platform -- \
  python -c "import asyncpg; asyncpg.connect('postgresql://orion:password@orion-postgres:5432/orion_db')"

# Check service endpoints
kubectl get endpoints -n orion-platform
```

#### 3. Ingress Not Working

```bash
# Check ingress controller
kubectl get pods -n ingress-nginx

# View ingress logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller

# Test DNS resolution
nslookup orion-platform.ai
```

### Performance Tuning

#### 1. Database Optimization

```sql
-- PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET random_page_cost = 1.1;
SELECT pg_reload_conf();
```

#### 2. Redis Configuration

```bash
# Set Redis memory policy
kubectl exec -it orion-redis-0 -n orion-platform -- \
  redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

#### 3. Application Optimization

```bash
# Increase worker processes
kubectl set env deployment/orion-api -n orion-platform WORKER_COUNT=8

# Adjust connection pools
kubectl set env deployment/orion-api -n orion-platform DATABASE_POOL_SIZE=50
```

## Production Checklist

- [ ] All secrets are properly configured
- [ ] SSL certificates are valid and auto-renewing
- [ ] Monitoring and alerting is operational
- [ ] Backup procedures are tested
- [ ] Disaster recovery plan is documented
- [ ] Resource limits are set appropriately
- [ ] Network policies are configured
- [ ] RBAC is properly configured
- [ ] Pod security policies are enforced
- [ ] Regular security updates are scheduled
- [ ] Load testing has been performed
- [ ] Documentation is up to date

## Support

For production support:
- Email: support@orion-platform.ai
- Slack: #orion-platform-ops
- On-call: Use PagerDuty integration

## License

Copyright (c) 2024 ORION Platform. All rights reserved.