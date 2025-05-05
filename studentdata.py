import streamlit as st
import pandas as pd
import joblib

# Load the trained model (make sure the model is in the same directory as the app)
model = joblib.load("student_model.pkl")

# Set the title of the app
st.title("🎓 Student Academic Performance Predictor")

# Display the model's accuracy (for example, you can hardcode it for now)
accuracy = 0.8607594936708861  # You can also calculate this dynamically if needed
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Input fields for student data (Grades, Study time, Failures, Absences)
G1 = st.slider("First Period Grade (G1)", 0, 20, 10)
G2 = st.slider("Second Period Grade (G2)", 0, 20, 10)
studytime = st.slider("Weekly Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)", 1, 4, 2)
failures = st.slider("Number of Past Class Failures", 0, 4, 0)
absences = st.slider("Number of School Absences", 0, 93, 10)

# When the button is clicked, make a prediction
if st.button("Predict Performance"):
    # Prepare input data in the form of a dataframe
    input_data = pd.DataFrame([[studytime, failures, absences, G1, G2]],
                              columns=["studytime", "failures", "absences", "G1", "G2"])
    
    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display the predicted performance
    st.success(f"Predicted Performance: {prediction}")
