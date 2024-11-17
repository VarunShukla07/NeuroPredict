import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO

# Load the trained model
rf_model = joblib.load('random_forest_model.pkl')

# Title and description
st.title("Alzheimer's Disease Prediction")
st.write("Fill in the details below to predict the likelihood of Alzheimer's Disease.")

# User inputs
st.header("Demographic Information")
age = st.number_input("Age", min_value=60, max_value=90, step=1)
gender = st.selectbox("Gender", options=["Male", "Female"])
ethnicity = st.selectbox("Ethnicity", options=["Caucasian", "African American", "Asian", "Other"])
education = st.selectbox("Education Level", options=["None", "High School", "Bachelor's", "Higher"])

st.header("Lifestyle & Clinical Factors")
bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, step=0.1)
smoking = st.checkbox("Smoking")
alcohol = st.slider("Alcohol Consumption (weekly units)", 0, 20, 0)
physical_activity = st.slider("Physical Activity (hours/week)", 0, 10, 0)
diet_quality = st.slider("Diet Quality (0-10)", 0, 10, 0)
sleep_quality = st.slider("Sleep Quality (4-10)", 4, 10, 0)

st.header("Medical History")
family_history = st.checkbox("Family History of Alzheimer's")
cardiovascular = st.checkbox("Cardiovascular Disease")
diabetes = st.checkbox("Diabetes")
depression = st.checkbox("Depression")
head_injury = st.checkbox("Head Injury")
hypertension = st.checkbox("Hypertension")

st.header("Clinical Measurements")
systolic_bp = st.slider("Systolic BP (mmHg)", 90, 180, 120)
diastolic_bp = st.slider("Diastolic BP (mmHg)", 60, 120, 80)
cholesterol_total = st.slider("Total Cholesterol (mg/dL)", 150, 300, 200)
cholesterol_ldl = st.slider("LDL Cholesterol (mg/dL)", 50, 200, 100)
cholesterol_hdl = st.slider("HDL Cholesterol (mg/dL)", 20, 100, 50)
cholesterol_triglycerides = st.slider("Triglycerides (mg/dL)", 50, 400, 150)

st.header("Cognitive Assessments")
mmse = st.slider("MMSE Score (0-30)", 0, 30, 15)
functional = st.slider("Functional Assessment Score (0-10)", 0, 10, 5)
adl = st.slider("ADL Score (0-10)", 0, 10, 5)

# Symptoms
st.header("Symptoms")
symptoms = {
    "Memory Complaints": st.checkbox("Memory Complaints"),
    "Behavioral Problems": st.checkbox("Behavioral Problems"),
    "Confusion": st.checkbox("Confusion"),
    "Disorientation": st.checkbox("Disorientation"),
    "Personality Changes": st.checkbox("Personality Changes"),
    "Difficulty Completing Tasks": st.checkbox("Difficulty Completing Tasks"),
    "Forgetfulness": st.checkbox("Forgetfulness")
}

# Prepare input for the model
symptom_values = [int(val) for val in symptoms.values()]
input_data = np.array([[
    age, int(gender == "Female"), ["Caucasian", "African American", "Asian", "Other"].index(ethnicity),
    ["None", "High School", "Bachelor's", "Higher"].index(education), bmi, int(smoking), alcohol,
    physical_activity, diet_quality, sleep_quality, int(family_history), int(cardiovascular),
    int(diabetes), int(depression), int(head_injury), int(hypertension), systolic_bp, diastolic_bp,
    cholesterol_total, cholesterol_ldl, cholesterol_hdl, cholesterol_triglycerides, mmse, functional,
    adl, *symptom_values
]])

# Prediction and output
if st.button("Predict"):
    prediction = rf_model.predict(input_data)
    probability = rf_model.predict_proba(input_data)[0][1] * 100
    result = "Alzheimer's Disease Detected" if prediction[0] == 1 else "No Alzheimer's Disease"
    
    st.write(f"Prediction: **{result}**")
    st.write(f"Likelihood: **{probability:.2f}%**")
    
    # Precautions
    if probability < 20:
        precautions = "Low risk. Maintain a healthy lifestyle and regular check-ups."
    elif probability < 50:
        precautions = "Moderate risk. Focus on physical activity, a balanced diet, and mental exercises."
    elif probability < 80:
        precautions = "High risk. Seek medical advice and consider cognitive therapy."
    else:
        precautions = "Very high risk. Immediate medical consultation is recommended."
    st.write(f"**Precautions:** {precautions}")

    # Visualizations
    st.subheader("Contributing Factors")
    categories = ['Lifestyle', 'Clinical', 'Cognitive']
    values = [
        physical_activity + diet_quality + sleep_quality + alcohol + (10 if smoking else 0),
        systolic_bp + diastolic_bp + cholesterol_total + cholesterol_ldl + cholesterol_hdl + cholesterol_triglycerides,
        mmse + functional + adl + sum(symptom_values)
    ]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff', '#99ff99'])
    ax.set_title("Factors Distribution")
    st.pyplot(fig)

    # Generate PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)  # Bold font for the title
    pdf.cell(0, 10, txt="Likelihood of Alzheimer", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Likelihood: {probability:.2f}%", ln=True)
    pdf.multi_cell(0, 10, txt=f"Precautions: {precautions}")

    # Save pie chart and add it to the PDF
    pie_chart_path = "pie_chart.png"
    fig.savefig(pie_chart_path)
    pdf.image(pie_chart_path, x=60, y=70, w=120)  # Move pie chart a little higher and center it
    pdf.ln(100)  # Add space after the pie chart for proper spacing

    # Generate a bar graph for Lifestyle, Clinical, Cognitive contributions
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(categories, values, color=['#ff9999', '#66b3ff', '#99ff99'])
    ax2.set_title("Contributions: Lifestyle vs Clinical vs Cognitive")
    ax2.set_ylabel("Total Value")
    ax2.grid(True)
    bar_chart_path = "bar_chart.png"
    fig2.savefig(bar_chart_path)
    plt.close(fig2)  # Close the plot to prevent overwriting
    pdf.image(bar_chart_path, x=60, y=180, w=120)  # Shift the bar chart down and center it

    # Add new page for the symptoms chart
    pdf.add_page()  
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    symptom_names = list(symptoms.keys())
    symptom_counts = list(symptoms.values())
    ax3.barh(symptom_names, symptom_counts, color='#ff9999')
    ax3.set_title("Symptoms Presence")
    ax3.set_xlabel("Presence (Yes=1, No=0)")
    ax3.grid(True, axis='x')
    symptoms_chart_path = "symptoms_chart.png"
    fig3.savefig(symptoms_chart_path)
    plt.close(fig3)  # Close the plot to prevent overwriting
    pdf.image(symptoms_chart_path, x=50, y=20, w=150)  # Proper placement of the third chart



    # Convert PDF to bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1')  # Returns content as bytes

    # Allow the user to download the report
    st.download_button(
        label="Download Report as PDF",
        data=pdf_bytes,
        file_name="Alzheimers_Prediction_Report.pdf",
        mime="application/pdf"
    )

