import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Load and preprocess data
data = pd.read_csv(r"C:\Users\manas\OneDrive\Desktop\diabetes_prediction_ml\diabetes_prediction_dataset.csv")
data["gender"] = data["gender"].map({"Male": 1, "Female": 2, "Other": 3})
data["smoking_history"] = data["smoking_history"].map({
    "never": 1, "No Info": 2, "current": 3, "former": 4, "ever": 5, "not current": 6
})

y = data['diabetes']
x = data.drop("diabetes", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

# Apply custom CSS styling
# st.markdown("""
#     <style>
#    /* styles.css */
# body {
#     font-family: 'Arial', sans-serif;
#     margin: 0;
#     padding: 0;
#     background-color: #f7f9fc;
#     color: #333;
# }

# header {
#     background-color: #4CAF50;
#     color: white;
#     padding: 20px 0;
#     text-align: center;
# }

# .header-container h1 {
#     margin: 0;
#     font-size: 2.5rem;
# }

# .header-container p {
#     font-size: 1.2rem;
#     margin-top: 10px;
# }

# main {
#     padding: 20px;
# }

# .form-section {
#     max-width: 600px;
#     margin: 0 auto;
#     background-color: white;
#     padding: 20px;
#     border-radius: 8px;
#     box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
# }

# .prediction-form .form-group {
#     margin-bottom: 15px;
# }

# .prediction-form label {
#     display: block;
#     font-weight: bold;
#     margin-bottom: 5px;
# }

# .prediction-form input,
# .prediction-form select {
#     width: 100%;
#     padding: 10px;
#     border: 1px solid #ccc;
#     border-radius: 4px;
# }

# .prediction-form .submit-button {
#     width: 100%;
#     background-color: #4CAF50;
#     color: white;
#     padding: 10px;
#     font-size: 1rem;
#     border: none;
#     border-radius: 4px;
#     cursor: pointer;
#     transition: background-color 0.3s;
# }

# .prediction-form .submit-button:hover {
#     background-color: #45a049;
# }

# .footer-container {
#             text-align: center;
#             margin-top: 50px;
#             padding: 10px;
#             background-color: #6200ea;
#             color: white;
#             border-radius: 8px;
# }
# .footer-container p {
#             margin: 0;
#             font-size: 0.9rem;
# }

#     </style>
#     """, unsafe_allow_html=True)

# # App layout and interactivity
# st.markdown(
#     """
#     <div class="header-container">
#         <h1>Diabetes Prediction</h1>
#         <p>Accurately Predict Diabetes Risk with Advanced Machine Learning</p>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# with st.form("prediction_form"):
#     st.markdown("### Enter Your Health Details")
    
#     gender = st.selectbox(
#         "Gender", 
#         ["Select Gender", "Male", "Female", "Other"], 
#         index=0, 
#         help="Select your gender."
#     )
#     age = st.number_input(
#         "Age", 
#         min_value=1, 
#         max_value=120, 
#         value=25, 
#         step=1, 
#         help="Enter your age in years."
#     )
#     hypertension = st.selectbox(
#         "Hypertension", 
#         ["No", "Yes"], 
#         index=0, 
#         help="Do you have hypertension?"
#     )
#     heart_disease = st.selectbox(
#         "Heart Disease", 
#         ["No", "Yes"], 
#         index=0, 
#         help="Do you have a heart disease history?"
#     )
#     smoking_history = st.selectbox(
#         "Smoking History", 
#         ["Select an Option", "Never", "No Info", "Current", "Former", "Ever", "Not Current"], 
#         index=0, 
#         help="Select your smoking history."
#     )
#     bmi = st.number_input(
#         "BMI", 
#         min_value=10.0, 
#         max_value=50.0, 
#         value=25.0, 
#         step=0.1, 
#         help="Enter your Body Mass Index."
#     )
#     hba1c = st.number_input(
#         "HbA1c Level", 
#         min_value=4.0, 
#         max_value=15.0, 
#         value=5.5, 
#         step=0.1, 
#         help="Enter your HbA1c Level (average blood glucose level over 3 months)."
#     )
#     blood_glucose = st.number_input(
#         "Blood Glucose Level", 
#         min_value=50, 
#         max_value=250, 
#         value=100, 
#         step=1, 
#         help="Enter your current Blood Glucose Level."
#     )
    
#     submit_button = st.form_submit_button(label="Predict")

# if submit_button:
#     # Map inputs to numeric values
#     gender_map = {"Male": 1, "Female": 2, "Other": 3}
#     smoking_map = {
#         "Never": 1, "No Info": 2, "Current": 3, 
#         "Former": 4, "Ever": 5, "Not Current": 6
#     }
    
#     # Check if a valid gender and smoking history option are selected
#     if gender == "Select Gender" or smoking_history == "Select an Option":
#         st.error("Please select valid options for Gender and Smoking History.")
#     else:
#         # Convert categorical inputs to numeric
#         gender_value = gender_map[gender]
#         smoking_value = smoking_map[smoking_history]
#         hypertension_value = 1 if hypertension == "Yes" else 0
#         heart_disease_value = 1 if heart_disease == "Yes" else 0

#         # Create input array for prediction
#         input_data = np.array([[age, gender_value, hypertension_value, 
#                                 heart_disease_value, smoking_value, 
#                                 bmi, hba1c, blood_glucose]])
        
#         # Placeholder for prediction model (replace this with actual model logic)
#         prediction = np.random.choice([0, 1])  # Dummy prediction for demonstration
#         result = "Diabetic." if prediction == 1 else "Not Diabetic."
#         color = "red" if prediction == 1 else "green"

#         st.markdown(
#             f"<p style='text-align:center; color:{color}; font-size:20px;'>{result}</p>", 
#             unsafe_allow_html=True
#         )
# st.markdown(
#     """
#     <div class="footer-container">
#         <p>© 2024 Diabetes Prediction App. All rights reserved.</p>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )



st.markdown("""
    <style>
    body {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(to right, #ece9e6, #ffffff);
        color: #333;
        margin: 0;
        padding: 0;
    }

    /* Header Container with Flex */
    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: #4CAF50;
        color: white;
        padding: 30px 0;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        text-align: center;
        margin-bottom: 20px;
    }

    .header-container h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .header-container p {
        font-size: 1.3rem;
        margin-top: 10px;
        font-weight: 300;
    }

    /* Form Container with Flex */
    .form-section {
        max-width: 700px;
        margin: 40px auto;
        padding: 30px;
        background-color: #fff;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        gap: 20px;  /* Adds space between form elements */
    }

    .form-group {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .stButton>button {
        margin: 0 auto;
        background-color: #4CAF50;
        color: white;
        padding: 12px 20px;
        font-size: 2rem;
        font-weight: bold;
        border: 2px solid blue;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s ease-in-out;
        width: 100%;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }

    /* Input fields styling */
    .form-section input, .form-section select {
        padding: 12px;
        border: 1px solid #ccc;
        border-radius: 8px;
        font-size: 1rem;
        transition: border-color 0.3s ease;
        
    }

    .form-section input:focus, .form-section select:focus {
        border-color: #4CAF50;
        outline: none;
        box-shadow: 0 0 8px rgba(76, 175, 80, 0.3);
    }

    /* Footer with Flex */
    .footer-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 30px;
        padding: 20px;
        background-color: #6200ea;
        color: white;
        border-radius: 12px;
        font-size: 0.9rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .stMarkdown p {
        
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# Form Layout in Streamlit
#  App layout and interactivity
st.markdown(
    """
    <div class="header-container">
        <h1>Diabetes Prediction</h1>
        <p>Accurately Predict Diabetes Risk with Advanced Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    st.markdown("### Enter Your Health Details")
    
    gender = st.selectbox(
        "Gender", 
        ["Select Gender", "Male", "Female", "Other"], 
        index=0, 
        help="Select your gender."
    )
    age = st.number_input(
        "Age", 
        min_value=1, 
        max_value=120, 
        value=25, 
        step=1, 
        help="Enter your age in years."
    )
    hypertension = st.selectbox(
        "Hypertension", 
        ["No", "Yes"], 
        index=0, 
        help="Do you have hypertension?"
    )
    heart_disease = st.selectbox(
        "Heart Disease", 
        ["No", "Yes"], 
        index=0, 
        help="Do you have a heart disease history?"
    )
    smoking_history = st.selectbox(
        "Smoking History", 
        ["Select an Option", "Never", "No Info", "Current", "Former", "Ever", "Not Current"], 
        index=0, 
        help="Select your smoking history."
    )
    bmi = st.number_input(
        "BMI", 
        min_value=10.0, 
        max_value=50.0, 
        value=25.0, 
        step=0.1, 
        help="Enter your Body Mass Index."
    )
    hba1c = st.number_input(
        "HbA1c Level", 
        min_value=4.0, 
        max_value=15.0, 
        value=5.5, 
        step=0.1, 
        help="Enter your HbA1c Level (average blood glucose level over 3 months)."
    )
    blood_glucose = st.number_input(
        "Blood Glucose Level", 
        min_value=50, 
        max_value=250, 
        value=100, 
        step=1, 
        help="Enter your current Blood Glucose Level."
    )
    
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    # Map inputs to numeric values
    gender_map = {"Male": 1, "Female": 2, "Other": 3}
    smoking_map = {
        "Never": 1, "No Info": 2, "Current": 3, 
        "Former": 4, "Ever": 5, "Not Current": 6
    }
    
    # Check if a valid gender and smoking history option are selected
    if gender == "Select Gender" or smoking_history == "Select an Option":
        st.error("Please select valid options for Gender and Smoking History.")
    else:
        # Convert categorical inputs to numeric
        gender_value = gender_map[gender]
        smoking_value = smoking_map[smoking_history]
        hypertension_value = 1 if hypertension == "Yes" else 0
        heart_disease_value = 1 if heart_disease == "Yes" else 0

        # Create input array for prediction
        input_data = np.array([[age, gender_value, hypertension_value, 
                                heart_disease_value, smoking_value, 
                                bmi, hba1c, blood_glucose]])
        
        # Placeholder for prediction model (replace this with actual model logic)
        prediction = np.random.choice([0, 1])  # Dummy prediction for demonstration
        result = "Diabetic." if prediction == 1 else "Not Diabetic."
        color = "red" if prediction == 1 else "green"

        st.markdown(
            
            f"<div style= 'background-color: #fbeaff'; border-radius: 4px><p style='text-align:center; color:{color}; font-size:30px;'>{result}</p></div>", 
            unsafe_allow_html=True
            
            
            
        )
st.markdown(
    """
    <div class="footer-container">
        <p>© 2024 Diabetes Prediction App. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
