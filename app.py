import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Set page config
st.set_page_config(page_title="Bengaluru House Price Predictor", page_icon="🏠")

# Title and description
st.title("🏠 Bengaluru House Price Predictor")
st.write("Predict house prices in Bengaluru based on various features")

# Load the dataset for getting unique values
@st.cache_data
def load_data():
    return pd.read_csv('Bengaluru_House_Data.csv')

dataset = load_data()

def convert_sqft(x):
    try:
        # Handle range values like '2100 - 2850'
        if '-' in str(x):
            tokens = x.split('-')
            return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2
        # Handle pure numbers
        return float(x)
    except:
        # For values like '34.46Sq. Meter', extract the number and convert to sqft if possible
        import re
        match = re.match(r"([\d\.]+)\s*Sq. Meter", str(x))
        if match:
            return float(match.group(1)) * 10.7639  # 1 sqm = 10.7639 sqft
        match = re.match(r"([\d\.]+)\s*Sq. Yards", str(x))
        if match:
            return float(match.group(1)) * 9.0  # 1 sq yard = 9 sqft
        match = re.match(r"([\d\.]+)\s*Acres", str(x))
        if match:
            return float(match.group(1)) * 43560  # 1 acre = 43560 sqft
        match = re.match(r"([\d\.]+)\s*Perch", str(x))
        if match:
            return float(match.group(1)) * 272.25  # 1 perch = 272.25 sqft
        match = re.match(r"([\d\.]+)\s*Guntha", str(x))
        if match:
            return float(match.group(1)) * 1089  # 1 guntha = 1089 sqft
        match = re.match(r"([\d\.]+)\s*Grounds", str(x))
        if match:
            return float(match.group(1)) * 2400  # 1 ground = 2400 sqft
        # If all fails, return NaN
        return np.nan

# Train the model once and cache it
@st.cache_resource
def train_model():
    # Prepare the training data
    X = dataset.iloc[:,:-1]
    y = dataset.iloc[:,-1]
    
    # Convert total_sqft
    X['total_sqft'] = X['total_sqft'].apply(convert_sqft)
    
    # Handle missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X.iloc[:, 5:8])  # columns 6 and 7: 'bath', 'balcony'
    X.iloc[:, 5:8] = imputer.transform(X.iloc[:, 5:8])
    
    # Handle categorical columns
    categorical_cols = ['area_type', 'availability', 'location', 'size', 'society']
    ct = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )
    X = ct.fit_transform(X)
    
    # Scale the data
    sc = StandardScaler(with_mean=False)
    X = sc.fit_transform(X)
    
    # Train the model
    regressor = RandomForestRegressor(n_estimators=18, random_state=0)
    regressor.fit(X, y)
    
    return regressor, ct, sc

# Train the model
model, column_transformer, scaler = train_model()

# Create input fields
st.subheader("Enter House Details")

# Get unique values for categorical columns
area_types = dataset['area_type'].unique()
locations = dataset['location'].unique()
sizes = dataset['size'].unique()

# Create input fields
area_type = st.selectbox("Area Type", area_types)
location = st.selectbox("Location", locations)
size = st.selectbox("Size (BHK)", sizes)
total_sqft = st.number_input("Total Square Feet", min_value=100.0, max_value=10000.0, value=1000.0)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
balcony = st.number_input("Number of Balconies", min_value=0, max_value=10, value=1)
availability = st.selectbox("Availability", dataset['availability'].unique())
society = st.selectbox("Society", dataset['society'].unique())

# Create a function to preprocess input data
def preprocess_input(area_type, location, size, total_sqft, bath, balcony, availability, society):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'area_type': [area_type],
        'availability': [availability],
        'location': [location],
        'size': [size],
        'society': [society],
        'total_sqft': [total_sqft],
        'bath': [bath],
        'balcony': [balcony]
    })
    
    # Convert total_sqft
    input_data['total_sqft'] = input_data['total_sqft'].apply(convert_sqft)
    
    # Transform the input data
    X = column_transformer.transform(input_data)
    
    # Scale the data
    X = scaler.transform(X)
    
    return X

# Add a predict button
if st.button("Predict Price"):
    try:
        # Preprocess the input data
        X = preprocess_input(area_type, location, size, total_sqft, bath, balcony, availability, society)
        
        # Make prediction
        predicted_price = model.predict(X)[0]
        
        # Display the prediction
        st.subheader("Predicted Price")
        st.write(f"₹{predicted_price:,.2f}L")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add some information about the model
st.sidebar.header("About")
st.sidebar.info(
    """
    This app predicts house prices in Bengaluru using a Random Forest Regressor model.
    The model takes into account various features such as:
    - Area Type
    - Location
    - Size (BHK)
    - Total Square Feet
    - Number of Bathrooms
    - Number of Balconies
    - Availability
    - Society
    """
) 