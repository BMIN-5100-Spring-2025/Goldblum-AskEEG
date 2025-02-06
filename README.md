# BMIN 5100 Template Repository
Template repository for BMIN 5100 Projects.

Contains a minimal Python project that reads a CSV from an input directory and
outputs a CSV to an output directory, suitable for an analysis workflow on Pennsieve.

Use this template to create your own GitHub repository by clicking the green
`Use this template` button towards the top-right of this page.

### Setup
Install the following:
- `python3` (latest)
- `pip` (or `pip3`, latest)

Then, run the following
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Running the application
```
python3 main/app.py
```

### Testing the application
```
pytest
```

## EEG Feature Processing
### Install system dependencies for YASA
#### On macOS:
```bash
brew install libomp
```

### Database Configuration
#### Add the following to a .env file:
```bash
DB_HOST=localhost
DB_PORT=<YOUR_DATABASE_PORT>
DB_NAME=<YOUR_DATABASE_NAME>
DB_USER=<YOUR_DATABASE_USERNAME>
DB_PASSWORD=<YOUR_DATABASE_PASSWORD>
```

#### Run the postgres container:
```bash
docker compose -f docker-compose.db.yml up -d
```