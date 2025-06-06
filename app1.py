import streamlit as st
import joblib
import pandas as pd

# Load model, preprocessor, and target scaler
model = joblib.load('model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
scaler = joblib.load('scaler.pkl')  # Scaler for target variable (price)

# Load dataset for dropdown options
dataset = pd.read_csv('Bengaluru_House_Data.csv')

# Set page configuration with a custom icon
st.set_page_config(page_title="Bengaluru House Price Predictor", page_icon="🏠", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 900px;
        }
        .stTitle {
            color: #2c3e50;
            font-family: 'Arial', sans-serif;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 1em;
        }
        .stSubheader {
            color: #3498db;
            font-family: 'Arial', sans-serif;
            text-align: center;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
            padding: 1.5rem;
        }
        .css-1aumxhk {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 0.5em 1.5em;
            font-size: 1em;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            display: block;
            margin: 1em auto;
        }
        .stButton>button:hover {
            background-color: #2980b9;
            border: none;
        }
        .stSuccess {
            background-color: #d4edda;
            color: #155724;
            border-radius: 5px;
            padding: 1em;
            text-align: center;
            font-size: 1.2em;
        }
        .stError {
            background-color: #f8d7da;
            color: #721c24;
            border-radius: 5px;
            padding: 1em;
            text-align: center;
        }
        .input-group {
            margin-bottom: 1em;
        }
        label {
            font-weight: bold;
            color: #2c3e50;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏠 Bengaluru House Price Predictor")
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Predict house prices in Bengaluru with ease using our machine learning model.</p>", unsafe_allow_html=True)

# Get unique values for dropdowns
area_types = dataset['area_type'].unique()
locations = dataset['location'].unique()
sizes = dataset['size'].unique()
availabilities = dataset['availability'].unique()
societies = dataset['society'].unique()

# Organize input fields in a structured layout using columns
st.markdown("### Enter Property Details")
with st.container():
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown("**Property Basics**")
        area_type = st.selectbox("Area Type", area_types, help="Select the type of area for the property")
        location = st.selectbox("Location", locations, help="Select the location of the property")
        size = st.selectbox("Size (BHK)", sizes, help="Select the size of the property (e.g., 2 BHK)")
        availability = st.selectbox("Availability", availabilities, help="Select the availability status")
    
    with col2:
        st.markdown("**Property Specifications**")
        total_sqft = st.number_input("Total Square Feet", min_value=100.0, max_value=10000.0, value=1000.0, step=10.0, help="Enter the total area in square feet")
        bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, help="Enter the number of bathrooms")
        balcony = st.number_input("Number of Balconies", min_value=0, max_value=10, value=1, help="Enter the number of balconies")
        society = st.selectbox("Society", societies, help="Select the society or community name")

# Function to extract numerical value from size
def extract_size(size_str):
    try:
        return int(size_str.split(' ')[0])
    except:
        return 0  # Fallback if parsing fails

# Predict button and result display
if st.button("Predict Price"):
    try:
        input_data = pd.DataFrame({
            'area_type': [area_type],
            'availability': [availability],
            'location': [location],
            'size': [extract_size(size)],  # Convert "2 BHK" to 2
            'society': [society],
            'total_sqft': [total_sqft],
            'bath': [bath],
            'balcony': [balcony]
        })
        X = preprocessor.transform(input_data)
        predicted_price = model.predict(X)[0]
        result = scaler.inverse_transform([[predicted_price]])[0][0]  # Inverse transform to get original scale
        result = max(0, result)  # Ensure price is not negative
        st.subheader("Predicted Price")
        st.success(f"₹ {result:,.2f} Lakhs")
        
        # Add a note if the price seems out of typical range (based on dataset)
        if result < 10 or result > 2000:
            st.markdown("<p style='text-align: center; color: #e67e22;'>Note: This prediction may be outside typical price ranges for Bengaluru houses (₹10 - ₹2000 Lakhs).</p>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Footer with additional information
st.markdown("---")
st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>Sachin Suresh©|All Rights Reserved</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>Data Source: Bengaluru House Data</p>", unsafe_allow_html=True)