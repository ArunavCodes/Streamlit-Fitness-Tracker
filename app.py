import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import datetime
import os

# -------------------------------
# Load ML models and scaler
# -------------------------------
calories_model = joblib.load('models/calories_model.pkl')
bmi_model = joblib.load('models/bmi_model.pkl')
heart_rate_model = joblib.load('models/heart_rate_model.pkl')
running_speed_model = joblib.load('models/running_speed_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("User Inputs")

age = st.sidebar.slider("Age", 10, 100, 25)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
height = st.sidebar.slider("Height (cm)", 100, 250, 170)
weight = st.sidebar.slider("Weight (kg)", 30, 150, 70)
running_time = st.sidebar.slider("Running Time (min)", 0, 120, 30)
running_speed = st.sidebar.slider("Running Speed (km/h)", 0, 40, 10)
distance = st.sidebar.slider("Distance (km)", 0, 50, 5)
heart_rate = st.sidebar.slider("Average Heart Rate (bpm)", 50, 200, 80)

st.sidebar.header("Fitness Goals")
goal_calories = st.sidebar.number_input("Goal Calories", 0, 5000, 500)
goal_steps = st.sidebar.number_input("Goal Steps", 0, 50000, 5000)
steps_taken = st.sidebar.number_input("Steps Taken", 0, 50000, 2000)

# -------------------------------
# Prediction Functions
# -------------------------------
def predict_calories():
    input_data = np.array([[age, 0 if gender=="Female" else 1, height, weight,
                            running_time, running_speed, distance, heart_rate]])
    input_data = scaler.transform(input_data)
    calories_burned = calories_model.predict(input_data)[0]
    return calories_burned

def calculate_bmi():
    h_m = height / 100
    bmi = weight / (h_m ** 2)
    return bmi

def calculate_workout_intensity():
    max_heart_rate = 220 - age
    intensity = (heart_rate / max_heart_rate) * 100
    return intensity

def calculate_running_speed():
    if running_time > 0:
        speed = distance / (running_time / 60)
        return speed
    else:
        return 0

def classify_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def provide_suggestions(bmi):
    if bmi < 18.5:
        return "Increase calorie intake and do strength training."
    elif 18.5 <= bmi < 25:
        return "Maintain current routine. Keep up the good work!"
    elif 25 <= bmi < 30:
        return "Incorporate more cardio and balanced diet."
    else:
        return "Consult a healthcare provider for a tailored fitness plan."

# -------------------------------
# Display Metrics
# -------------------------------
st.title("SMARTFIT AI - Personal Fitness Tracker")

calories_burned = predict_calories()
bmi = calculate_bmi()
intensity = calculate_workout_intensity()
speed = calculate_running_speed()
bmi_category = classify_bmi(bmi)
suggestion = provide_suggestions(bmi)

st.subheader("Your Fitness Metrics")
st.write(f"**Calories Burned:** {calories_burned:.2f}")
st.write(f"**BMI:** {bmi:.2f} ({bmi_category})")
st.write(f"**Workout Intensity:** {intensity:.2f}%")
st.write(f"**Running Speed:** {speed:.2f} km/h")
st.write(f"**Fitness Suggestion:** {suggestion}")

# -------------------------------
# Track Progress
# -------------------------------
if goal_calories > 0:
    calories_progress = (calories_burned / goal_calories) * 100
else:
    calories_progress = 0

if goal_steps > 0:
    steps_progress = (steps_taken / goal_steps) * 100
else:
    steps_progress = 0

st.subheader("Progress")
st.write(f"**Calories Progress:** {calories_progress:.2f}%")
st.write(f"**Steps Progress:** {steps_progress:.2f}%")

# -------------------------------
# Show Data Analysis
# -------------------------------
try:
    df = pd.read_csv('calories_burned_data.csv')
    st.subheader("Calories Burned Distribution")
    st.bar_chart(df['Calories Burned'])
except:
    st.warning("Dataset 'calories_burned_data.csv' not found for plotting.")

# -------------------------------
# Compare with Dataset Averages
# -------------------------------
if st.button("Compare with Dataset Averages"):
    try:
        avg_calories_burned = df['Calories Burned'].mean()
        avg_bmi = (df['Weight(kg)'] / ((df['Height(cm)'] / 100) ** 2)).mean()
        avg_running_speed = df['Running Speed(km/h)'].mean()
        avg_heart_rate = df['Average Heart Rate'].mean()

        st.subheader("Comparison with Dataset Averages")
        st.write(f"**Your Calories Burned:** {calories_burned:.2f} | Dataset Avg: {avg_calories_burned:.2f}")
        st.write(f"**Your BMI:** {bmi:.2f} | Dataset Avg: {avg_bmi:.2f}")
        st.write(f"**Your Running Speed:** {speed:.2f} | Dataset Avg: {avg_running_speed:.2f}")
        st.write(f"**Your Heart Rate:** {heart_rate:.2f} | Dataset Avg: {avg_heart_rate:.2f}")
    except:
        st.warning("Dataset not available for comparison.")

# -------------------------------
# Generate PDF Report
# -------------------------------
if st.button("Generate PDF Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="SMARTFIT AI - Fitness Report", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.cell(200, 10, txt="------------------------------", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, txt=f"Height: {height} cm", ln=True)
    pdf.cell(200, 10, txt=f"Weight: {weight} kg", ln=True)
    pdf.cell(200, 10, txt=f"Calories Burned: {calories_burned:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"BMI: {bmi:.2f} ({bmi_category})", ln=True)
    pdf.cell(200, 10, txt=f"Workout Intensity: {intensity:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Running Speed: {speed:.2f} km/h", ln=True)
    pdf.cell(200, 10, txt=f"Fitness Suggestion: {suggestion}", ln=True)
    pdf.output("fitness_report.pdf")
    st.success("PDF report generated as fitness_report.pdf")
