# BMIN 5100 Project: AskEEG

## Environment Variables

### Copy the .env.example file to .env and fill in the necessary values:
```bash
cp .env.example .env
```

## Running EEG Synchrony Analysis

### 1. Run Locally with Docker
```bash
# Build the Docker image
docker build -t goldblum_askeeg:v1 .

# Run in local mode (no S3 interaction)
docker run -v $(pwd)/data:/data goldblum_askeeg:v1
```

### 2. Run Using AWS Fargate
```bash
# Alternatively, run with S3 integration for testing
docker run -v ~/.aws:/root/.aws \
  -e RUN_MODE=fargate \
  -e S3_BUCKET_NAME=goldblum-askeeg \
  goldblum_askeeg:v1
```

For Fargate execution, the application will automatically:
- Download input data from S3 bucket (`data/input/` prefix)
- Process the EEG data
- Upload results to S3 bucket (`data/output/` prefix)

The ECS task definition is configured with necessary environment variables.

## AWS Setup Instructions

### Upload data to AWS S3
```bash
# Upload EDF file for Fargate processing
aws s3 cp data/input/EMU1371_Day02_1_5006_to_5491.edf s3://goldblum-askeeg/data/input/EMU1371_Day02_1_5006_to_5491.edf
```

### Push Docker Image to ECR
1. Get account ID and region information
```bash
aws sts get-caller-identity

# Returns:
# "UserId": "<USER-ID>",
# "Account": "<ACCOUNT>",
# "Arn": "<ARN>"
```

2. Build the Docker image
```bash
docker build -t goldblum_askeeg:v1 .
```

3. Authenticate Docker to ECR
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com
```

4. Tag Docker image for ECR
```bash
docker tag goldblum_askeeg:v1 <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/goldblum_askeeg:v1
```

5. Push to ECR
```bash
docker push <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/goldblum_askeeg:v1
```