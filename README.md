# Rising Waters Flood Prediction System

## Project Overview

Rising Waters is a production-oriented Flood Prediction System foundation created for an internship project. This repository currently contains the base Flask application structure and supporting project directories.

Machine learning, prediction logic, frontend pages, styling, JavaScript, and notebooks are intentionally not implemented at this stage.

## Folder Structure

```text
rising-waters-flood-prediction/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
├── notebooks/
├── models/
├── templates/
├── static/
│   ├── css/
│   └── js/
└── screenshots/
```

### Directory Purpose

- `data/`: Stores raw, processed, or external datasets for future flood prediction work.
- `notebooks/`: Reserved for exploratory analysis and experimentation notebooks.
- `models/`: Reserved for trained model artifacts and serialized files.
- `templates/`: Reserved for future Flask HTML templates.
- `static/css/`: Reserved for future stylesheet files.
- `static/js/`: Reserved for future JavaScript files.
- `screenshots/`: Stores application screenshots, reports, or visual project evidence.

## Installation

1. Clone the repository.

```bash
git clone <repository-url>
cd rising-waters-flood-prediction
```

2. Create and activate a virtual environment.

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

## Running Instructions

Run the Flask application:

```bash
python app.py
```

Open the local development URL shown in the terminal, usually:

```text
http://127.0.0.1:5000/
```
