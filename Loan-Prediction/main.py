import pickle
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pandas as pd

# Define custom CSS styles
custom_styles = """
    <style>
        body {
            background-color: #f0f0f0;
            font-family: Arial, sans-serif;
        }
        .header {
            background-color: #333;
            padding: 15px;
            text-align: center;
        }
        h1 {
            color: #fff;
        }
        .input-container {
            margin-top: 20px;
            padding: 20px;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .result {
            font-size: 24px;
            margin-top: 20px;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }
        .accepted {
            background-color: #4CAF50;
            color: #fff;
        }
        .rejected {
            background-color: #FF5733;
            color: #fff;
        }
    </style>
"""

st.markdown(custom_styles, unsafe_allow_html=True)

def Logistic_model():
    train = pd.read_csv('loan_data.csv')
    # Categorical to numerical
    train['Gender'] = train['Gender'].map({'Male': 0, 'Female': 1})
    train['Married'] = train['Married'].map({'No': 0, 'Yes': 1})
    train['Loan_Status'] = train['Loan_Status'].map({'N': 0, 'Y': 1})

    # Separating dependent and independent variables
    X = train[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount']]
    y = train.Loan_Status

    # Training the decision tree classifier model
    model = DecisionTreeClassifier()
    model.fit(X, y)

    pickle_out = open("classifier_dt.pkl", mode="wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()

# This is the main function in which we define our app  
def main():       
    # Header of the page 
    st.markdown("<div class='header'><h1>Check Your Loan Eligibility</h1></div>", unsafe_allow_html=True)

    # Create input containers
    with st.container():
        Gender = st.selectbox('Gender', ("Male", "Female", "Other"))
        Married = st.selectbox('Marital Status', ("Unmarried", "Married", "Other"))
        ApplicantIncome = st.number_input("Monthly Income in Rupees")
        LoanAmount = st.number_input("Loan Amount in Rupees")

    result = ""

    # When 'Check' is clicked, make the prediction and display it
    if st.button("Check"):
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount)
        st.markdown(f"<div class='result {result.lower()}'>{result}</div>", unsafe_allow_html=True)

# Prediction using the data which the user inputs 
def prediction(Gender, Married, ApplicantIncome, LoanAmount):
    
    # Loading the trained model
    pickle_in = open('classifier_dt.pkl', 'rb')
    classifier = pickle.load(pickle_in)

    # Pre-processing the data 
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1

    if Married == "Married":
        Married = 1
    else:
        Married = 0

    prediction = classifier.predict([[Gender, Married, ApplicantIncome, LoanAmount]])
    
    if prediction == 1:
        pred = 'Loan Accepted'
    else:
        pred = 'Loan Rejected'
    return pred

if __name__ == '__main__':
    Logistic_model()
    main()
