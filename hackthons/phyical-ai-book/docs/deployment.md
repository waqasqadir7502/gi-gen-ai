# Deployment Instructions for Physical AI Book RAG Chatbot

This document provides instructions for deploying the Physical AI Book RAG Chatbot to various platforms.

## Platform Options

The application can be deployed to various platforms including:
- Vercel (recommended for frontend)
- Railway (recommended for backend)
- AWS
- Google Cloud Platform
- Azure
- Self-hosted servers

## Prerequisites

- Valid Cohere API key
- Qdrant Cloud account (or self-hosted Qdrant instance)
- Backend API key for authentication
- Domain name (optional but recommended)

## Environment Variables

Before deployment, ensure the following environment variables are set:

### Backend Environment Variables
````
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
BACKEND_API_KEY=your_secure_backend_api_key
```

### Frontend Environment Variables (if needed)
````
REACT_APP_BACKEND_API_KEY=your_backend_api_key
REACT_APP_API_BASE_URL=https://your-backend-domain.com
```

## Deployment to Vercel (Frontend)

### Prerequisites
- Vercel account
- `vercel` CLI installed (`npm install -g vercel`)

### Steps
1. Navigate to the project root directory
2. Set environment variables in Vercel dashboard or CLI:
   ```bash
   vercel env add COHERE_API_KEY production
   vercel env add QDRANT_URL production
   vercel env add QDRANT_API_KEY production
   vercel env add BACKEND_API_KEY production
   ```
3. Deploy:
   ```bash
   vercel --prod
   ```

## Deployment to Railway (Backend)

### Prerequisites
- Railway account
- `railway` CLI installed

### Steps
1. Create a new Railway project
2. Link your local repository:
   ```bash
   railway link
   ```
3. Set environment variables:
   ```bash
   railway var set COHERE_API_KEY=your_cohere_api_key
   railway var set QDRANT_URL=your_qdrant_url
   railway var set QDRANT_API_KEY=your_qdrant_api_key
   railway var set BACKEND_API_KEY=your_secure_backend_api_key
   ```
4. Deploy:
   ```bash
   railway up
   ```

## Deployment with Docker

### Prerequisites
- Docker installed
- Docker Compose (optional)

### Steps
1. Create a Dockerfile for the backend:
   ```dockerfile
   FROM python:3.10-slim

   WORKDIR /app

   COPY backend/requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8000

   CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
2. Build and run the container:
   ```bash
   docker build -t physical-ai-book-backend .
   docker run -d -p 8000:8000 \
     -e COHERE_API_KEY=your_cohere_api_key \
     -e QDRANT_URL=your_qdrant_url \
     -e QDRANT_API_KEY=your_qdrant_api_key \
     -e BACKEND_API_KEY=your_secure_backend_api_key \
     --name physical-ai-book-backend \
     physical-ai-book-backend
   ```

## Deployment with PM2 (Self-hosted)

### Prerequisites
- Node.js installed
- PM2 installed (`npm install -g pm2`)

### Steps
1. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. Create an ecosystem file `ecosystem.config.js`:
   ```javascript
   module.exports = {
     apps: [{
       name: 'physical-ai-book-backend',
       script: 'uvicorn',
       args: 'main:app --host 0.0.0.0 --port 8000',
       interpreter: 'python',
       cwd: './backend',
       env: {
         NODE_ENV: 'production',
         COHERE_API_KEY: 'your_cohere_api_key',
         QDRANT_URL: 'your_qdrant_url',
         QDRANT_API_KEY: 'your_qdrant_api_key',
         BACKEND_API_KEY: 'your_secure_backend_api_key'
       }
     }]
   };
   ```
3. Start the application:
   ```bash
   pm2 start ecosystem.config.js
   pm2 startup
   pm2 save
   ```

## Post-Deployment Steps

### Initial Content Indexing
After deployment, you need to index your documentation:

1. SSH into your deployed backend or access the deployment console
2. Run the indexing script:
   ```bash
   cd backend
   python scripts/ingest_docs.py --docs-path ../docs
   ```

### Health Checks
Verify that your deployment is working:
- Access the health endpoint: `GET /health`
- Test the chat endpoint: `POST /api/chat`
- Verify that the frontend can communicate with the backend

### Configuration Verification
- Ensure API keys are correctly configured
- Verify that rate limiting is functioning
- Confirm security headers are applied
- Test error handling and logging

## Scaling Recommendations

### Horizontal Scaling
- Use multiple instances behind a load balancer
- Implement sticky sessions if needed
- Scale based on request volume and response time

### Vertical Scaling
- Increase instance resources (CPU/RAM) based on demand
- Monitor resource utilization
- Scale up before reaching limits

## Monitoring and Maintenance

### Health Monitoring
- Set up uptime monitoring for your endpoints
- Monitor response times and error rates
- Implement alerts for service degradation

### Performance Monitoring
- Track API response times
- Monitor database query performance
- Watch resource utilization

### Security Monitoring
- Regular security scans
- Monitor for unusual access patterns
- Review and rotate API keys periodically

## Troubleshooting

### Common Issues
- **Connection timeouts**: Verify network connectivity to external services
- **Authentication failures**: Check API keys and their validity
- **Rate limiting**: Monitor request patterns and adjust as needed
- **Memory issues**: Increase instance memory or optimize code

### Debugging
- Enable detailed logging in development/staging
- Check application logs for errors
- Verify environment variable configuration
- Test connectivity to external services

## Rollback Procedures

If issues arise after deployment:
1. Keep previous versions ready for rollback
2. Use platform-specific rollback features
3. Have database migration rollback plans
4. Communicate with users about service status

For support, refer to the respective platform documentation or contact the development team.