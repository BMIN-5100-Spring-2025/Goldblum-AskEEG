# BMIN 5100 Project: AskEEG

## Environment Variables

### Copy the .env.example file to .env and fill in the necessary values:
```bash
cp .env.example .env
```

## Running EEG Synchrony Analysis

### 1. Run Using Local Data
```bash
# Build the Docker image
docker build -t goldblum_askeeg:latest .

# Run in local mode (no S3 interaction)
docker run -v $(pwd)/data:/data goldblum_askeeg:latest
```

### 2. Run Using AWS S3 Data
```bash
# Alternatively, run with S3 integration for testing
docker run -v ~/.aws:/root/.aws \
  -e RUN_MODE=fargate \
  -e S3_BUCKET_NAME=goldblum-askeeg \
  goldblum_askeeg:latest
```

For Fargate execution, the application will automatically:
- Download input data from S3 bucket (`data/input/` prefix)
- Process the EEG data
- Upload results to S3 bucket (`data/output/` prefix)

The ECS task definition is configured with necessary environment variables.

### 3. Run on AWS Fargate
Once the infrastructure is deployed with Terraform and the Docker image is pushed to ECR (steps detailed below in "AWS Setup Instructions"), the task can be run directly using the AWS CLI:

```bash
# Get the required values from Terraform
cd terraform
TASK_DEFINITION_ARN=$(terraform output -raw ecs_task_definition_arn)

# Get remote infrastructure values
CLUSTER_ARN=$(terraform state show data.terraform_remote_state.infrastructure | grep ecs_cluster_arn | awk '{print $3}' | tr -d '"')
PRIVATE_SUBNET=$(terraform state show data.terraform_remote_state.infrastructure | grep private_subnet_id | awk '{print $3}' | tr -d '"')
SECURITY_GROUP=$(terraform state show data.terraform_remote_state.infrastructure | grep ecs_security_group_id | awk '{print $3}' | tr -d '"')

# Run the ECS task
aws ecs run-task \
  --cluster $CLUSTER_ARN \
  --task-definition $TASK_DEFINITION_ARN \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$PRIVATE_SUBNET],securityGroups=[$SECURITY_GROUP],assignPublicIp=DISABLED}"
```

## AWS Setup Instructions

### Upload data to AWS S3
```bash
# Upload the EDF file to AWS S3
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
# Build for x86_64 environment
docker buildx build --platform linux/amd64 -t goldblum_askeeg:latest .
```

3. Authenticate Docker to ECR
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com
```

4. Tag Docker image for ECR
```bash
docker tag goldblum_askeeg:latest <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/goldblum_askeeg:latest
```

5. Push to ECR
```bash
docker push <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/goldblum_askeeg:latest
```