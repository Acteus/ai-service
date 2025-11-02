# Fly.io Deployment Guide for ISO 21001 AI Service

## Overview

This guide provides step-by-step instructions for deploying the Flask AI Service to Fly.io. The service includes 8 machine learning models for ISO 21001 compliance prediction and educational analytics.

## Prerequisites

1. **Fly.io Account**: Sign up at [fly.io](https://fly.io)
2. **Fly CLI**: Install the Fly CLI tool
   ```bash
   # macOS
   brew install flyctl

   # Linux
   curl -L https://fly.io/install.sh | sh

   # Windows (PowerShell)
   iwr https://fly.io/install.ps1 -useb | iex
   ```

3. **Trained Models**: Ensure ML models are trained and saved in the `models/` directory
   ```bash
   python train_models.py
   ```

## Project Structure for Deployment

```
ai-service/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Optimized for Fly.io
├── fly.toml                # Fly.io configuration
├── .gitignore              # Git ignore patterns
├── ai_models/              # ML model classes
├── utils/                  # Utility functions
├── data/                   # Training data (not deployed)
├── models/                 # Trained ML models (deployed)
└── logs/                   # Application logs (not deployed)
```

## Deployment Steps

### Step 1: Authenticate with Fly.io

```bash
fly auth login
```

### Step 2: Initialize Fly App (First Time Only)

```bash
# Navigate to the ai-service directory
cd ai-service

# Launch the app (this creates fly.toml if it doesn't exist)
fly launch --name iso21001-ai-service --region sin

# Answer prompts:
# - Choose organization: [your-org]
# - Would you like to copy its configuration to the new app: No
# - Do you want to deploy now: No (we'll deploy manually)
```

### Step 3: Configure Environment Variables

Create environment secrets for sensitive configuration:

```bash
# Set Flask environment
fly secrets set FLASK_DEBUG=false
fly secrets set FLASK_APP=app.py
fly secrets set PYTHONUNBUFFERED=1

# Optional: Set API keys if needed
fly secrets set FLASK_AI_API_KEY=your-optional-api-key

# Set Laravel integration URL (update with your actual Laravel app URL)
fly secrets set LARAVEL_BASE_URL=https://your-laravel-app.fly.dev
```

### Step 4: Deploy the Application

```bash
# Deploy to Fly.io
fly deploy

# Monitor deployment logs
fly logs
```

### Step 5: Verify Deployment

```bash
# Check app status
fly status

# Get the deployed URL
fly info

# Test health endpoint
curl https://your-app-name.fly.dev/health

# Test a prediction endpoint
curl -X POST https://your-app-name.fly.dev/api/v1/compliance/predict \
  -H "Content-Type: application/json" \
  -d '{
    "learner_needs_index": 4.2,
    "satisfaction_score": 3.8,
    "success_index": 4.1,
    "safety_index": 4.5,
    "wellbeing_index": 3.9,
    "overall_satisfaction": 4.0
  }'
```

## Configuration Details

### Fly.io Configuration (`fly.toml`)

- **Region**: Singapore (`sin`) - closest to Philippines
- **Memory**: 1GB RAM (sufficient for ML models)
- **CPU**: 1 shared CPU core
- **Auto-scaling**: Enabled with connection-based scaling
- **Health Checks**: HTTP health check on `/health` endpoint
- **Persistent Storage**: Optional volume for model updates

### Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `FLASK_PORT` | `8080` | Port for the Flask app |
| `FLASK_DEBUG` | `false` | Disable debug mode in production |
| `FLASK_APP` | `app.py` | Main application file |
| `PYTHONUNBUFFERED` | `1` | Ensure Python output is not buffered |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LARAVEL_BASE_URL` | URL | Laravel app URL for CORS |

### Docker Configuration

The Dockerfile is optimized for Fly.io:
- Uses Python 3.11 slim base image
- Includes Gunicorn for production serving
- Health checks configured
- Multi-stage build for smaller image size

## Scaling and Performance

### Resource Allocation

- **Memory**: 1GB (adequate for ML model loading)
- **CPU**: 1 shared core (sufficient for concurrent requests)
- **Concurrency**: 25 connections max, 20 soft limit

### Auto-scaling

Fly.io automatically scales based on:
- CPU usage
- Memory usage
- Request queue length
- Connection count

### Monitoring

```bash
# View app metrics
fly metrics

# View logs
fly logs

# Check app status
fly status

# Monitor resource usage
fly vm status
```

## Model Management

### Initial Model Deployment

Models are included in the Docker image during build. For updates:

1. **Retrain models locally**:
   ```bash
   python train_models.py
   ```

2. **Redeploy**:
   ```bash
   fly deploy
   ```

### Persistent Model Storage (Optional)

For frequent model updates, configure persistent volume:

```bash
# Create volume
fly volumes create ai_service_data --size 10

# Mount in fly.toml (already configured)
[mounts]
  source = "ai_service_data"
  destination = "/data"
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   ```bash
   # Check if models directory exists
   fly ssh console
   ls -la models/
   ```

2. **Memory Issues**:
   ```bash
   # Check memory usage
   fly metrics
   # Increase memory if needed
   fly scale memory 2048
   ```

3. **Health Check Failures**:
   ```bash
   # Check health endpoint
   curl https://your-app.fly.dev/health
   # View logs
   fly logs
   ```

4. **CORS Issues**:
   - Update `LARAVEL_BASE_URL` in environment variables
   - Ensure Laravel app URL is correct

### Logs and Debugging

```bash
# View recent logs
fly logs

# Stream logs in real-time
fly logs -f

# SSH into running instance
fly ssh console

# Check running processes
fly ps
```

## Cost Optimization

### Free Tier Limits
- 512MB RAM (upgrade to 1GB for ML models)
- 1 shared CPU
- 100GB outbound data/month

### Paid Tier Recommendations
- **Memory**: 1GB minimum for ML models
- **CPU**: 1 dedicated CPU for better performance
- **Scaling**: Enable auto-scaling for variable loads

## Laravel Integration

### Update Laravel Configuration

After deployment, update your Laravel `.env`:

```env
# Flask AI Service Configuration
FLASK_AI_SERVICE_URL=https://your-app-name.fly.dev
FLASK_AI_API_KEY=your-optional-api-key
AI_TIMEOUT_SECONDS=30
AI_MAX_RETRIES=3
AI_ENABLE_CACHE=true
AI_FALLBACK_TO_PHP=true
```

### Test Integration

```bash
# From Laravel project root
php artisan ai:test-flask
```

## Security Considerations

1. **Environment Variables**: Use `fly secrets` for sensitive data
2. **API Keys**: Rotate regularly using `fly secrets set`
3. **CORS**: Properly configured for Laravel integration
4. **Health Checks**: Monitor service availability
5. **Logs**: Don't expose sensitive information in logs

## Backup and Recovery

### Model Backups
```bash
# Download models from running instance
fly ssh console
tar -czf models_backup.tar.gz models/
fly ssh sftp get /app/models_backup.tar.gz
```

### Configuration Backup
```bash
# Backup fly.toml
cp fly.toml fly.toml.backup
```

## Support

For issues:
1. Check Fly.io documentation: https://fly.io/docs
2. Review application logs: `fly logs`
3. Test endpoints manually with `curl`
4. Check model loading in SSH console

---

**Last Updated**: November 2025
**Fly.io Version**: Latest
**Python Version**: 3.11