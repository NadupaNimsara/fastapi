import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import os

# Define the path to the pickle file (relative path)
model_path = 'trained_model.sav'

# Check if the file exists and load the model
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
else:
    st.error(f"Model file '{model_path}' not found.")
    loaded_model = None

# Creating a function for Prediction
def diabetes_prediction(input_data):
    if loaded_model is None:
        return "Model not loaded. Please check the file path."
    
    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # Title for the main page
    st.title('WELCOME TO THE DIABETES PREDICTION SYSTEM')

    # Selectbox for different sections
    st.sidebar.title("Dashboard")

    option = st.sidebar.selectbox('Select a section:', 
                          ('Home', 'Prediction', 'Visualization'))

    # Section 1: Introduction
    if option == 'Home':
        st.header("Introduction")
        st.write("""
Welcome to the Diabetes Prediction System! This application is designed to help predict whether a person is diabetic based on various medical parameters. 
Our prediction model is built using advanced machine learning techniques, providing you with quick and reliable results.

To get started, navigate to the 'Prediction' section, input the necessary details, and receive your prediction instantly. 
For insights and trends, explore the 'Visualization' section to see how different factors relate to diabetes.

Your health is important, and this tool is here to assist you in better understanding the risks associated with diabetes.
""")
        st.image("Diabetes.jpg", use_column_width=True)

    # Section 2: User input for prediction
    elif option == 'Prediction':
        st.header("Input for Prediction")
        
        # Getting the input data from the user
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1, step=1)
        Glucose = st.number_input('Glucose Level', min_value=0, max_value=300, value=120)
        BloodPressure = st.number_input('Blood Pressure value', min_value=0, max_value=200, value=70)
        SkinThickness = st.number_input('Skin Thickness value', min_value=0, max_value=100, value=20)
        Insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=80)
        BMI = st.number_input('BMI value', min_value=0.0, max_value=70.0, value=25.0)
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=2.5, value=0.5)
        Age = st.number_input('Age of the Person', min_value=0, max_value=120, value=30)
        
        # Button for prediction
        if st.button('Diabetes Test Result'):
            diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
            st.success(diagnosis)

    # Section 3: Data Visualization
    elif option == 'Visualization':
        st.header("Data Visualization")

        # Histogram: Distribution of Glucose Levels
        if st.checkbox('Show Glucose Level Histogram'):
            glucose_levels = np.random.randint(80, 200, size=100)
            plt.figure(figsize=(10, 5))
            plt.hist(glucose_levels, bins=20, color='green', edgecolor='black')
            plt.title('Histogram of Glucose Levels')
            plt.xlabel('Glucose Level')
            plt.ylabel('Frequency')
            st.pyplot(plt)
            plt.clf()  # Clear the figure to avoid overlap in subsequent plots

        # Example: Visualizing Glucose Level vs Age
        if st.checkbox('Show Glucose vs Age Graph'):
            # Sample data
            ages = np.random.randint(20, 70, size=100)
            glucose_levels = np.random.randint(80, 200, size=100)

            plt.figure(figsize=(10, 5))
            plt.scatter(ages, glucose_levels, color='blue')
            plt.title('Glucose Level vs Age')
            plt.xlabel('Age')
            plt.ylabel('Glucose Level')
            st.pyplot(plt)
            plt.clf()  # Clear the figure to avoid overlap in subsequent plots

        # Additional graphs can be added here

if __name__ == '__main__':
    main()
