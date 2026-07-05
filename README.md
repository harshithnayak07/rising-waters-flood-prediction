# Rising Waters Flood Prediction System

## Project Overview

Rising Waters Flood Prediction System is a machine learning-based flood prediction web application built using Flask and XGBoost. The system analyzes weather and rainfall-related parameters, preprocesses the data, compares multiple machine learning models, and uses the best-performing model for flood risk prediction through a responsive web interface.

This project is designed to be beginner-friendly while still following a clean end-to-end machine learning workflow: data analysis, preprocessing, model training, model comparison, model saving, and Flask deployment.

## Features

- Flood prediction using weather parameters
- Data analysis and visualization
- Data preprocessing
- Multiple ML models
- Model comparison
- XGBoost selected as final model
- Flask web application
- Responsive frontend

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- HTML
- CSS
- JavaScript
- Joblib

## Project Structure

```text
rising-waters-flood-prediction/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   `-- flood_dataset.xlsx
|-- models/
|   |-- floods.save
|   `-- transform.save
|-- notebooks/
|   `-- FloodPrediction.ipynb
|-- src/
|   `-- train_models.py
|-- templates/
|   |-- home.html
|   |-- index.html
|   |-- chance.html
|   `-- no_chance.html
|-- static/
|   |-- css/
|   |   `-- main.css
|   `-- js/
|       `-- main.js
`-- screenshots/
```

## Machine Learning Workflow

```text
Dataset
|
v
EDA
|
v
Preprocessing
|
v
Feature Scaling
|
v
Decision Tree
|
v
Random Forest
|
v
KNN
|
v
XGBoost
|
v
Model Comparison
|
v
Save Model
|
v
Flask Deployment
```

## Installation

1. Clone the repository.

```bash
git clone <repository-url>
cd rising-waters-flood-prediction
```

2. Create a virtual environment.

```bash
python -m venv venv
```

3. Activate the virtual environment.

On Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

4. Install the required dependencies.

```bash
pip install -r requirements.txt
```

## Running the Project

1. Train and save the final model if the saved model is not already available.

```bash
python src/train_models.py
```

2. Start the Flask application.

```bash
python app.py
```

3. Open the application in your browser.

```text
http://127.0.0.1:5000/
```

## Application Screenshots

### Home Page

Add screenshot here:

```text
screenshots/home_page.png
```

### Prediction Form

Add screenshot here:

```text
screenshots/prediction_form.png
```

### Flood Result

Add screenshot here:

```text
screenshots/flood_result.png
```

### No Flood Result

Add screenshot here:

```text
screenshots/no_flood_result.png
```

## Future Improvements

- User authentication
- Database integration
- Live weather API
- Cloud deployment
