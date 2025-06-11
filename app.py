import streamlit as st
import joblib

# Load your model
model = joblib.load('D:/Personal/titanic/Gradient_boosting.pkl')

st.title("Titanic Survival Predictor")

# Get user input for each feature
pclass = st.selectbox('Passenger Class (Pclass):', [1, 2, 3])
sex = st.selectbox('Sex (0=male, 1=female):', [0, 1])
age = st.number_input('Age:', 0, 100, 25)
familysize = st.number_input('Family Size:', 1, 20, 1)
fare = st.number_input('Fare:', 0.0, 600.0, 30.0)
deck = st.number_input('Deck:', 0, 8, 0)
embarked_q = st.selectbox('Embarked Q:', [0, 1])
# embarked_s = st.selectbox('Embarked S:', [0, 1])

if st.button('Predict'):
    features = [[pclass, sex, age, familysize, fare, deck, embarked_q]]
    prediction = model.predict(features)
    st.write("Prediction:", "Survived!" if prediction[0] == 1 else "Did not survive.")
