import pandas as pd
import streamlit as st
import joblib
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AgglomerativeClustering  # Import scikit-learn
# Assuming `data` is your original dataset used for clustering

# Load the saved KMeans model and scaler
kmeans = joblib.load("kmeans__model.joblib")
scaler = joblib.load("scaler.joblib")

# Define the features for user input
features = [
    'BirthRate', 'BusinessTaxRate', 'CO2Emissions', 'DaystoStartBusiness',
    'EaseofBusiness', 'EnergyUsage', 'GDP', 'Health Exp GDP',
    'HourstodoTax', 'InfantMortalityRate', 'InternetUsage',
    'LendingInterest', 'LifeExpectancyFemale', 'LifeExpectancyMale',
    'MobilePhoneUsage', 'PopulationTotal', 'TourismInbound',
    'TourismOutbound', 'Country'
]

# Define the prediction function with scaling
def predict_cluster(new_data):
    new_data_df = pd.DataFrame([new_data])
    new_data_df = new_data_df[features]  # Keep only relevant features
    # Scale the new data using the loaded scaler
    new_data_scaled = scaler.transform(new_data_df)
    cluster =  kmeans.predict(new_data_df)[0]
    return cluster

# Function to provide cluster insights
def cluster_insights(cluster):
    if cluster == 0:
        return (
            "Cluster 0: These countries generally have a low business tax rate where "
            "the ease of doing business varies widely. As tax rates increase, the ease of "
            "doing business stabilizes. Other characteristics:\n"
            "- **Birth Rate**: Moderate to high\n"
            "- **CO2 Emissions**: Moderate levels\n"
            "- **Ease of Business**: High variability\n"
            "- **Tourism Inbound**: Moderate to high"
        )
    elif cluster == 1:
        return (
            "Cluster 1: Countries in this cluster typically have a low business tax rate, "
            "leading to a mix of high and low ease of doing business scores. When tax rates rise, "
            "the ease of business stabilizes at high levels. Other characteristics:\n"
            "- **Health Expenditure (% GDP)**: High\n"
            "- **Mobile Phone Usage**: High\n"
            "- **Internet Usage**: High\n"
            "- **Ease of Business**: Generally stable around 90"
        )
    elif cluster == 2:
        return (
            "Cluster 2: These countries maintain high ease of doing business scores even with "
            "both low and high tax rates, although they are fewer in number. Other characteristics:\n"
            "- **Life Expectancy (Female & Male)**: High\n"
            "- **Energy Usage**: Moderate\n"
            "- **Tourism Inbound and Outbound**: Moderate\n"
            "- **Population Total**: Smaller populations"
        )
    else:
        return "Unknown cluster type."

# Streamlit user interface
st.title("KMeans Clustering Prediction with Loaded Model")
st.write("Adjust the sliders for the following features:")

# Create slider inputs for each feature with defined ranges
user_input = {}
user_input['BirthRate'] = st.slider("BirthRate", 0.0, 0.1, 0.02)
user_input['BusinessTaxRate'] = st.slider("BusinessTaxRate", 0.0, 50.0, 20.0)
user_input['CO2Emissions'] = st.slider("CO2Emissions", 0.0, 200000.0, 10000.0)
user_input['DaystoStartBusiness'] = st.slider("DaystoStartBusiness", 0.0, 365.0, 15.0)
user_input['EaseofBusiness'] = st.slider("EaseofBusiness", 0.0, 100.0, 80.0)
user_input['EnergyUsage'] = st.slider("EnergyUsage", 0.0, 50000.0, 15000.0)
user_input['GDP'] = st.slider("GDP", 0.0, 5e12, 1e10)
user_input['Health Exp GDP'] = st.slider("Health Exp GDP", 0.0, 1.0, 0.05)
user_input['HourstodoTax'] = st.slider("HourstodoTax", 0.0, 500.0, 200.0)
user_input['InfantMortalityRate'] = st.slider("InfantMortalityRate", 0.0, 100.0, 30.0)
user_input['InternetUsage'] = st.slider("InternetUsage", 0.0, 100.0, 70.0)
user_input['LendingInterest'] = st.slider("LendingInterest", 0.0, 20.0, 5.0)
user_input['LifeExpectancyFemale'] = st.slider("LifeExpectancyFemale", 0.0, 100.0, 75.0)
user_input['LifeExpectancyMale'] = st.slider("LifeExpectancyMale", 0.0, 100.0, 70.0)
user_input['MobilePhoneUsage'] = st.slider("MobilePhoneUsage", 0.0, 100.0, 90.0)
user_input['PopulationTotal'] = st.slider("PopulationTotal", 0.0, 1e9, 5e6)
user_input['TourismInbound'] = st.slider("TourismInbound", 0.0, 1e8, 1e5)
user_input['TourismOutbound'] = st.slider("TourismOutbound", 0.0, 1e8, 5e4)
user_input['Country'] = st.slider("Country", 1, 200, 1)

# Predict button
if st.button("Predict Cluster"):
    predicted_cluster = predict_cluster(user_input)
    st.write(f"The predicted cluster for the input is: {predicted_cluster}")
    st.write(cluster_insights(predicted_cluster))
