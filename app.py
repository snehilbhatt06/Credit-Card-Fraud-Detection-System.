import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Create input fields for user to enter feature values
input_df = st.text_input('Input All features')

# Modified part: Clean and split the input features
input_df_lst = [x.strip().strip('"') for x in input_df.split(',')]

# Create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    try:
        # Ensure all input values are converted to float and handle empty strings
        features = np.array([float(x) for x in input_df_lst if x], dtype=np.float64)

        # Check if the number of features is correct
        if len(features) == X_train.shape[1]:
            # Make prediction
            prediction = model.predict(features.reshape(1, -1))
            # Display result
            if prediction[0] == 0:
                st.write("Legitimate transaction")
            else:
                st.write("Fraudulent transaction")
        else:
            st.write(f"Error: Expected {X_train.shape[1]} features, but got {len(features)}")
    except ValueError as e:
        st.write(f"Error: {e}")
