# BMIN 5100 Project: AskEEG

### Create a virtual environment and install Python dependencies
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Install system dependencies for YASA
#### On macOS:
```bash
brew install libomp
```

### Configure the database
#### Add the following to a .env file:
```bash
DB_HOST=localhost
DB_PORT=<YOUR_DATABASE_PORT>
DB_NAME=<YOUR_DATABASE_NAME>
DB_USER=<YOUR_DATABASE_USERNAME>
DB_PASSWORD=<YOUR_DATABASE_PASSWORD>
```

### Run the database container
```bash
docker compose -f docker-compose.db.yml up -d
```

### Write EDF data to the database
```bash
python3 app/edf_to_postgres.py
```

### Run the YASA algorithm on the data
```bash
python3 app/yasa_from_postgres.py
```
*Output:*
```
YASA predictions from PostgreSQL data:
C3: ['W' 'W' 'W' 'R' 'R' 'R' 'W' 'W' 'W' 'W' 'W' 'W' 'W' 'W' 'W' 'W']
Cz: ['N2' 'N2' 'N2' 'N1' 'N2' 'N1' 'W' 'W' 'W' 'W' 'W' 'N1' 'R' 'R' 'R' 'R']
C4: ['W' 'W' 'W' 'W' 'N2' 'R' 'R' 'W' 'W' 'W' 'W' 'W' 'W' 'W' 'W' 'W']
Consensus: ['W', 'W', 'W', nan, 'N2', 'R', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W']
```

### Upload data to AWS S3
```bash
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
docker build -t <IMAGE>:<TAG> .
# docker build -t goldblum_askeeg:v1 .
```

3. Authenticate Docker to ECR
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com
```

4. Tag Docker image for ECR
```bash
docker tag goldblum_askeeg:v1 <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/<IMAGE>:<TAG>
# docker tag goldblum_askeeg:v1 <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/goldblum_askeeg:v1
```

5. Push to ECR
```bash
docker push <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/<IMAGE>:<TAG>
# docker push <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/goldblum_askeeg:v1
```