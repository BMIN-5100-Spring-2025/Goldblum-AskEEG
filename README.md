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