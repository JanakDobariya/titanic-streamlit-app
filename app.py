import streamlit as st
import joblib

st.title("Titanic Survival Predictor")
st.write(
    "This app predicts whether a passenger would have survived the Titanic disaster based on their characteristics.")

try:
    # Load your trained Gradient Boosting model
    model = joblib.load('Gradient_boosting.pkl')
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    st.stop()

# --- Passenger Info ---
st.subheader("Passenger Information")
pclass = st.selectbox('Passenger Class:', [1, 2, 3],
                      help="1 = First Class, 2 = Second Class, 3 = Third Class")
sex = st.selectbox('Sex:', ['Male', 'Female'],
                   help="Select the passenger's gender")
sex = 1 if sex == 'Female' else 0  # Match model encoding

age = st.number_input('Age:', 0, 100, 25,
                      help="Enter the passenger's age")
familysize = st.number_input('Family Size:', 1, 20, 1,
                             help="Total number of family members traveling together")
fare = st.number_input('Fare (in pounds):', 0.0, 600.0, 30.0,
                       help="Ticket price in pounds")
deck = st.number_input('Cabin:', 0, 8, 0,
                       help="Cabin number (0-8)")

# --- Embarked Info ---
embarked_choice = st.radio(
    'Select port of embarkation',
    ['Cherbourg', 'Queenstown', 'Southampton']
)

# One-hot encoding for Embarked
embarked_C = 1 if embarked_choice == 'Cherbourg' else 0
embarked_Q = 1 if embarked_choice == 'Queenstown' else 0
embarked_S = 1 if embarked_choice == 'Southampton' else 0

# --- Prediction ---
if st.button('Predict Survival'):
    try:
        # Make sure the order matches training
        features = [[pclass, sex, age, familysize, fare, deck, embarked_C, embarked_Q, embarked_S]]
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0]

        st.subheader("Prediction Results")
        if prediction[0] == 1:
            st.success("Survived! 🎉")
        else:
            st.error("Did not survive 😢")

        st.write(f"Probability of survival: {probability[1]:.2%}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
