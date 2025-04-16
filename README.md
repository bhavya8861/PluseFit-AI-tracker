# PulseFit-AI-tracker

PulseFit AI Tracker is an innovative online platform designed to help you monitor and improve your health and fitness. Through its intuitive website, you can effortlessly track a variety of key fitness metrics such as heart rate, steps, calories burned, sleep quality, and much more, all powered by advanced AI technology.

The platform provides personalized insights and real-time feedback based on your activity data. Whether you're tracking your workouts, managing your progress, or setting new fitness goals, PulseFit AI Tracker uses smart algorithms to tailor recommendations to your specific needs, ensuring you get the most out of your fitness journey.

With easy-to-navigate dashboards, detailed progress reports, and the ability to sync with your devices, the website allows you to access your fitness data anytime, anywhere. PluseFit AI Tracker isn’t just about tracking; it’s about optimizing your health and fitness experience using AI-powered insights that evolve with you.

Perfect for anyone seeking a reliable, data-driven approach to fitness, PluseFit AI Tracker's website empowers you to make smarter decisions and achieve your wellness goals more efficiently.
Download all the file to get started!
The provided file (app.py) is a Streamlit-based web application for a fitness tracking and calorie burn prediction tool called FitGenius AI. It includes features like user registration, login, input parameter collection (e.g., age, BMI, exercise duration), and a machine learning model (Random Forest Regressor) to predict calories burned. It also provides personalized recommendations and a simple AI chatbot for fitness-related queries.

How to Run the Application
Prerequisites:

Install Python (3.7 or higher).
Install required libraries:
pip install streamlit pandas numpy scikit-learn
Ensure you have the following files in the same directory as app.py:
calories.csv (dataset for calories burned).
exercise.csv (dataset for exercise data).
Background images (e.g., login img.jpeg, after login img.jpg).
Run the Application:

Open a terminal or command prompt.
Navigate to the directory containing app.py.
Run the following command:
streamlit run app.py
The app will start, and a local URL (e.g., http://localhost:8501) will open in your browser.
Using the App:

Register: Create a new account by providing a username and password.
Login: Use your credentials to log in.
Input Parameters: Enter your fitness details (age, BMI, exercise duration, etc.) in the sidebar.
Predict Calories: View the predicted calories burned and personalized recommendations.
Features
User Registration & Login: Secure authentication system.
Calorie Burn Prediction: Predicts calories burned based on user inputs (age, BMI, exercise duration, etc.).
Personalized Recommendations: AI-powered suggestions to improve fitness.
AI Chatbot: Answers fitness-related questions.
Interactive UI: Dynamic background images and user-friendly interface.
How to Run
Install dependencies:
pip install streamlit pandas numpy scikit-learn
Download the required datasets (calories.csv and exercise.csv) and place them in the same directory as app.py.
Run the app:
streamlit run app.py
Open your browser and navigate to http://localhost:8501.
Technologies Used
Python: Core programming language.
Streamlit: For building the web app.
Pandas & NumPy: For data manipulation.
Scikit-Learn: For machine learning (Random Forest Regressor).
Dataset
calories.csv: Contains calorie burn data.
exercise.csv: Contains exercise-related data.
License
This project is licensed under the MIT License. See LICENSE for details.


## **Notes**:
- Replace `https://example.com/path-to-your-logo.png` and other placeholder URLs with actual paths to your images or screenshots.
- Ensure the datasets (`calories.csv` and `exercise.csv`) are included in your repository or provide instructions on how to obtain them.
About fitness_tracker file!
This Jupyter Notebook file (fitness_tracker.ipynb) is a data analysis project focused on fitness data, utilizing datasets from calories.csv and exercise.csv. The notebook performs data exploration, visualization, and preparation for machine learning tasks. It begins by loading and merging the datasets, which contain user information such as age, height, weight, exercise duration, heart rate, body temperature, and calories burned. The notebook then explores the data through descriptive statistics and visualizations like boxplots and heatmaps to identify patterns and check for null values. It prepares the data for machine learning by splitting it into training and testing sets and scaling the features. Finally, a linear regression model is built to predict calories burned based on the provided features. The notebook serves as a comprehensive guide for analyzing fitness data and building predictive models.

About requirements.txt file!
The requirements.txt file is a text document used in Python projects to list all the dependencies (libraries and packages) required for the project to run. Each line in the file specifies a package name along with its version number, ensuring that the project uses compatible versions of the libraries. This file is crucial for setting up a consistent development environment, as it allows developers to easily install all necessary dependencies using package managers like pip. The provided requirements.txt includes a wide range of packages, such as Flask for web development, pandas for data manipulation, numpy for numerical computations, torch for machine learning, and many others, indicating that the project likely involves web development, data analysis, and machine learning tasks.

About Users.csv file!
The users.csv file is a Comma-Separated Values (CSV) file that stores user credentials, specifically usernames and passwords. Each row in the file represents a user, with the first column containing the username and the second column containing the corresponding password. This type of file is commonly used in applications to manage user authentication data. However, storing passwords in plain text, as seen in this file, is highly insecure and not recommended. Best practices suggest using hashed passwords and additional security measures to protect sensitive information.
