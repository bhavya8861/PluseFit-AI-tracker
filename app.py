import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings
import base64

warnings.filterwarnings('ignore')

# === Set custom background ===
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def add_bg_image():
    # Option 1: Online image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b"); 
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Option 2: Local image — Uncomment below and add your image file
    # img_base64 = get_base64_image("your_local_image.jpg")
    # st.markdown(
    #     f"""
    #     <style>
    #     .stApp {{
    #         background-image: url("data:image/jpg;base64,{img_base64}");
    #         background-size: cover;
    #         background-attachment: fixed;
    #     }}
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )

add_bg_image()

# === App Title ===
st.write("## 🏃‍♂️ PulseFit AI Tracker")
st.write("Predict how many **calories you'll burn** during exercise using parameters like `Age`, `Gender`, `Height`, `Weight`, etc.")

# === Sidebar input ===
st.sidebar.header("🔧 User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    height = st.sidebar.slider("Height (cm): ", 120, 220, 170)
    weight = st.sidebar.slider("Weight (kg): ", 40, 150, 70)
    duration = st.sidebar.slider("Duration (min): ", 0, 60, 30)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 180, 90)
    body_temp = st.sidebar.slider("Body Temperature (°C): ", 35.0, 42.0, 37.5, step=0.1)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0
    bmi = round(weight / ((height / 100) ** 2), 2)

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

# === Show user input ===
st.write("---")
st.header("📋 Your Parameters:")
st.dataframe(df)

# === Load and process data ===
try:
    calories = pd.read_csv("C:/Users/Yogesh Gowda/Downloads/PulseFit/CSV files/calories.csv")
    exercise = pd.read_csv("C:/Users/Yogesh Gowda/Downloads/PulseFit/CSV files/exercise.csv")
except FileNotFoundError:
    st.error("❌ Required dataset files not found. Make sure 'calories.csv' and 'exercise.csv' exist.")
    st.stop()

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

# Calculate BMI
exercise_df["BMI"] = round(exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2), 2)

# Model data preparation
model_data = exercise_df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
model_data = pd.get_dummies(model_data, drop_first=True)

train_data, test_data = train_test_split(model_data, test_size=0.2, random_state=42)
X_train = train_data.drop("Calories", axis=1)
y_train = train_data["Calories"]

# Train model
model = RandomForestRegressor(n_estimators=500, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Predict
df = df.reindex(columns=X_train.columns, fill_value=0)

st.write("---")
st.header("🎯 Predicted Calories Burned")
with st.spinner("Calculating..."):
    prediction = model.predict(df)
    time.sleep(1)

st.success(f"🔥 Estimated Calories Burned: **{round(prediction[0], 2)} kilocalories**")

# === Similar results ===
st.write("---")
st.header("📊 Similar Results")
cal_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[
    (exercise_df["Calories"] >= cal_range[0]) & 
    (exercise_df["Calories"] <= cal_range[1])
]
st.dataframe(similar_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]].sample(5))

# === General stats ===
st.write("---")
st.header("📈 How You Compare")

st.write(f"You are older than **{round((exercise_df['Age'] < df['Age'].values[0]).mean() * 100, 2)}%** of users.")
st.write(f"Your exercise duration is longer than **{round((exercise_df['Duration'] < df['Duration'].values[0]).mean() * 100, 2)}%** of users.")
st.write(f"Your heart rate is higher than **{round((exercise_df['Heart_Rate'] < df['Heart_Rate'].values[0]).mean() * 100, 2)}%** of users.")
st.write(f"Your body temperature is higher than **{round((exercise_df['Body_Temp'] < df['Body_Temp'].values[0]).mean() * 100, 2)}%** of users.")

# === AI Chatbot ===
st.write("---")
st.header("🤖 AI Fitness Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.chat_message("assistant").write("Hi! I'm FitBot 🤖. Ask me anything about calories, BMI, or workouts!")

user_input = st.chat_input("Type your question here...")

if user_input:
    user_message = user_input
    st.session_state.messages.append({"role": "user", "content": user_message})

    # Simple keyword-based responses
    response = ""
    if "calorie" in user_message.lower():
        response = "Calories are estimated based on your weight, duration, and intensity of exercise."
    elif "bmi" in user_message.lower():
        response = "BMI is calculated as weight (kg) divided by height (m²). A healthy BMI is between 18.5 and 24.9."
    elif "heart rate" in user_message.lower():
        response = "Heart rate reflects exercise intensity. A higher heart rate means more calories burned."
    else:
        response = "I'm still learning! Ask me more about BMI, calories, or workouts."
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display the conversation
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

