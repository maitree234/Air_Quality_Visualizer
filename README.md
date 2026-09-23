# Air Quality Visualizer

A Python-based web application for monitoring air quality using live data from the Open-Meteo Air Quality API.

The application retrieves pollutant concentrations for selected Indian cities, calculates an Indian AQI-based value, visualizes pollutant trends, and uses machine learning models to predict the next-hour PM2.5 concentration.

## Features

- Live air-quality data using REST API
- City selection for Indian cities
- PM2.5, PM10, NO₂, SO₂, O₃ and NH₃ monitoring
- Indian AQI-based calculation
- AQI category classification
- Dominant pollutant identification
- 24-hour pollutant trend visualization
- 24-hour AQI trend visualization
- PM2.5 prediction using Machine Learning
- Linear Regression and Random Forest comparison
- Mean Absolute Error (MAE) evaluation
- Random Forest feature importance
- CSV data download
- Interactive Streamlit dashboard

## Technologies Used

- Python
- Streamlit
- Pandas
- Requests
- Plotly
- Scikit-learn
- Open-Meteo Air Quality REST API

## Machine Learning

Two regression models are used for PM2.5 prediction:

1. Linear Regression
2. Random Forest Regression

The models use features such as:

- Hour of day
- Day of week
- Previous-hour PM2.5
- Previous-hour PM10
- Previous-hour NO₂

Model performance is evaluated using Mean Absolute Error (MAE).

## Data Processing

The application:

1. Sends latitude and longitude to the Open-Meteo Air Quality API.
2. Receives pollutant data in JSON format.
3. Converts the JSON response into a Pandas DataFrame.
4. Processes pollutant concentrations.
5. Calculates AQI-based values.
6. Displays the results using interactive Plotly charts.
7. Uses the processed data for machine learning.

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL