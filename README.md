# Bengaluru House Price Predictor

A machine learning model that predicts house prices in Bengaluru based on various features such as area type, location, size, and more.

## Features

- Predicts house prices in Bengaluru
- Takes into account multiple features:
  - Area Type
  - Location
  - Size (BHK)
  - Total Square Feet
  - Number of Bathrooms
  - Number of Balconies
  - Availability
  - Society

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy

## Setup and Installation

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Dataset

The model is trained on the Bengaluru House Data dataset, which contains information about house prices in Bengaluru.

## Model

The model uses a Random Forest Regressor with the following preprocessing steps:
- Area unit conversion
- Missing value imputation
- Categorical variable encoding
- Feature scaling 