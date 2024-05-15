from flask import Flask, render_template, request, redirect, url_for, session
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from datetime import datetime, timedelta
from dateutil.parser import parse
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Google Fit API settings
SCOPES = [
    'https://www.googleapis.com/auth/userinfo.profile',
    'https://www.googleapis.com/auth/user.birthday.read',
    'https://www.googleapis.com/auth/fitness.body.read',
    'https://www.googleapis.com/auth/fitness.oxygen_saturation.read',
    'https://www.googleapis.com/auth/fitness.activity.read',
    'https://www.googleapis.com/auth/fitness.blood_pressure.read',
    'https://www.googleapis.com/auth/fitness.activity.read',
    'https://www.googleapis.com/auth/fitness.heart_rate.read'
]
API_VERSION = 'v1'
API_USER_ID = 'me'

# Load datasets
training_data = "/home/rudransh/new new new new new/templates/Training.csv"
testing_data = "/home/rudransh/new new new new new/templates/Testing.csv"

# Load data
data = pd.read_csv(training_data).dropna(axis=1)
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Initialize models
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X_train, y_train)
final_nb_model.fit(X_train, y_train)
final_rf_model.fit(X_train, y_train)

# Define symptom index
symptoms = X.columns.values
symptom_index = {symptom.capitalize(): index for index, symptom in enumerate(symptoms)}

# Define prediction function
def predict_disease(symptoms):
    input_data = [0] * len(symptom_index)
    for symptom in symptoms:
        index = symptom_index.get(symptom.capitalize())
        if index is not None:
            input_data[index] = 1
    input_data = np.array(input_data).reshape(1, -1)

    rf_prediction = encoder.inverse_transform([final_rf_model.predict(input_data)])[0]
    nb_prediction = encoder.inverse_transform([final_nb_model.predict(input_data)])[0]
    svm_prediction = encoder.inverse_transform([final_svm_model.predict(input_data)])[0]

    rf_accuracy = accuracy_score(y_test, final_rf_model.predict(X_test))
    nb_accuracy = accuracy_score(y_test, final_nb_model.predict(X_test))
    svm_accuracy = accuracy_score(y_test, final_svm_model.predict(X_test))

    rf_cm = confusion_matrix(y_test, final_rf_model.predict(X_test))
    nb_cm = confusion_matrix(y_test, final_nb_model.predict(X_test))
    svm_cm = confusion_matrix(y_test, final_svm_model.predict(X_test))

    # Combined model predictions
    combined_predictions = [rf_prediction, nb_prediction, svm_prediction]
    final_prediction = np.unique(combined_predictions)[0]

    # Concatenate predictions for calculating accuracy score
    combined_predictions_all = np.concatenate([final_rf_model.predict(X_test).reshape(-1, 1),
                                               final_nb_model.predict(X_test).reshape(-1, 1),
                                               final_svm_model.predict(X_test).reshape(-1, 1)], axis=1)
    combined_predictions_all = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=combined_predictions_all)

    # Combined model accuracy
    combined_accuracy = accuracy_score(y_test, combined_predictions_all)

    # Combined model confusion matrix
    combined_cm = confusion_matrix(y_test, combined_predictions_all)

    return {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction,
        "rf_accuracy": rf_accuracy,
        "nb_accuracy": nb_accuracy,
        "svm_accuracy": svm_accuracy,
        "rf_cm": rf_cm,
        "nb_cm": nb_cm,
        "svm_cm": svm_cm,
        "combined_accuracy": combined_accuracy,
        "combined_cm": combined_cm
    }

def save_confusion_matrix(matrix, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

def save_prediction_to_file(symptoms, predictions):
    file_name = f"output/{','.join(symptoms)}.txt"
    with open(file_name, "w") as file:
        file.write(f"Symptoms: {', '.join(symptoms)}\n")
        file.write(f"Random Forest Prediction: {predictions['rf_model_prediction']}\n")
        file.write(f"Naive Bayes Prediction: {predictions['naive_bayes_prediction']}\n")
        file.write(f"SVM Prediction: {predictions['svm_model_prediction']}\n")
        file.write(f"Final Prediction: {predictions['final_prediction']}\n")
        file.write(f"Random Forest Accuracy: {predictions['rf_accuracy']:.2f}\n")
        file.write(f"Naive Bayes Accuracy: {predictions['nb_accuracy']:.2f}\n")
        file.write(f"SVM Accuracy: {predictions['svm_accuracy']:.2f}\n")
        file.write("Random Forest Confusion Matrix:\n")
        file.write(str(predictions['rf_cm']) + "\n")
        file.write("Naive Bayes Confusion Matrix:\n")
        file.write(str(predictions['nb_cm']) + "\n")
        file.write("SVM Confusion Matrix:\n")
        file.write(str(predictions['svm_cm']) + "\n")
        file.write("Combined Model Confusion Matrix:\n")
        file.write(str(predictions['combined_cm']) + "\n")

    return file_name

@app.route('/', methods=['GET', 'POST'])
def index():
    global symptoms
    if request.method == 'POST':
        # Initialize symptoms if not already initialized
        if symptoms is None:
            symptoms = X.columns.values
        selected_symptoms = [request.form[f'symptom_{i}'] for i in range(1, 7)]
        predictions = predict_disease(selected_symptoms)
        file_name = save_prediction_to_file(selected_symptoms, predictions)
        
        # Save confusion matrix images
        save_confusion_matrix(predictions['rf_cm'], "static/rf_cm.png")
        save_confusion_matrix(predictions['nb_cm'], "static/nb_cm.png")
        save_confusion_matrix(predictions['svm_cm'], "static/svm_cm.png")
        save_confusion_matrix(predictions['combined_cm'], "static/combined_cm.png")
        
        return render_template('prediction.html', symptoms=selected_symptoms, predictions=predictions, file_name=file_name)

    if symptoms is None:
        symptoms = X.columns.values
    return render_template('index.html')

@app.route('/previous_predictions')
def previous_predictions():
    # Get list of files in the 'output' directory
    file_list = os.listdir("output")
    # Pass file_list to the template
    return render_template('previous_predictions.html', file_list=file_list)

# Define route to view a specific file
@app.route('/view_file/<file_name>')
def view_file(file_name):
    # Construct the file path
    file_path = f"output/{file_name}"
    # Read the file content
    with open(file_path, "r") as file:
        file_content = file.read()
    # Return the file content as a response
    return file_content

@app.route('/authenticate', methods=['GET', 'POST'])
def authenticate_google_fit():
    if request.method == 'POST':
        try:
            data = get_google_data()
            return render_template('google_fit_data.html', data=data)
        except Exception as e:
            return f"Error fetching Google Fit data: {e}"
    else:
        # Handle GET request, for example, redirect to home page
        return redirect(url_for('home'))

def get_google_clients():
    """Get the Google Fit and People services with the stored credentials."""
    creds = None
    if 'access_token' in session:
        creds_data = {
            'token': session['access_token'],
            'scopes': SCOPES,
            'expiry': session['expiry']
        }
        creds = Credentials.from_authorized_user_info(info=creds_data)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_config(
            json.load(open('credentials.json')), SCOPES)
        creds = flow.run_local_server(port=8000)
        session['access_token'] = creds.token
        session['refresh_token'] = creds.refresh_token
        session['token_uri'] = creds.token_uri
        session['client_id'] = creds.client_id
        session['client_secret'] = creds.client_secret
        session['scopes'] = creds.scopes
        session['expiry'] = creds.expiry
    fit_client = build('fitness', 'v1', credentials=creds)
    people_client = build('people', 'v1', credentials=creds)
    return fit_client, people_client

def get_google_data():
    """Fetch the user's Google Fit and People data."""
    fit_client, people_client = get_google_clients()

    # Fetch profile data from Google People API
    profile = people_client.people().get(resourceName='people/me', personFields='names,ageRanges').execute()
    name = profile.get('names', [{}])[0].get('displayName', 'Unknown')
    age = profile.get('ageRanges', [{}])[0].get('ageRange', 'Unknown')

    # Fetch Google Fit data
    now = datetime.now().isoformat("T") + "Z"  # 'Z' indicates UTC time
    seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat("T") + "Z"

    # Weight data request
    weight_data_request = {
        "aggregateBy": [{
            "dataTypeName": "com.google.weight",
            "dataSourceId": "derived:com.google.weight:com.google.android.gms:merge_weight"
        }],
        "bucketByTime": { "durationMillis": 86400000 },  # Duration is in milliseconds, 86400000ms = 1 day
        "startTimeMillis": int(parse(seven_days_ago).timestamp() * 1000),
        "endTimeMillis": int(parse(now).timestamp() * 1000)
    }
    weight_data = fit_client.users().dataset().aggregate(userId='me', body=weight_data_request).execute()
    buckets = weight_data.get('bucket', [])
    weight = 'Unknown'
    if buckets:
        datasets = buckets[-1].get('dataset', [])
        if datasets:
            points = datasets[0].get('point', [])
            if points:
                values = points[0].get('value', [])
                if values:
                    weight = f"{round(values[0].get('fpVal', 'Unknown'), 1)} kg"
    else:
        weight = "No weight data available"

    # Height data request
    height_data_request = {
        "aggregateBy": [{
            "dataTypeName": "com.google.height",
            "dataSourceId": "derived:com.google.height:com.google.android.gms:merge_height"
        }],
        "bucketByTime": { "durationMillis": 86400000 },
        "startTimeMillis": int(parse(seven_days_ago).timestamp() * 1000),
        "endTimeMillis": int(parse(now).timestamp() * 1000)
    }
    height_data = fit_client.users().dataset().aggregate(userId='me', body=height_data_request).execute()
    buckets = height_data.get('bucket', [])
    height = 'Unknown'
    if buckets:
        datasets = buckets[-1].get('dataset', [])
        if datasets:
            points = datasets[0].get('point', [])
            if points:
                values = points[0].get('value', [])
                if values:
                    height = f"{round(values[0].get('fpVal', 'Unknown'), 2)} m"
    else:
        height = "No height data available"

    # Sleep data request

    # Calculate start and end times for the past week
    seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat("T") + "Z"
    now = datetime.now().isoformat("T") + "Z"

    # Initialize sleep duration
    sleep_duration = "No sleep data available"

    # Sleep data request for the past week
    sleep_dataset = fit_client.users().sessions().list(userId='me', fields='session', startTime=seven_days_ago, endTime=now).execute()

    if sleep_dataset.get('session'):
        for session in sleep_dataset['session']:
            activity_type = session.get("activityType")
            if activity_type == 72:  # Update this value if necessary
                sleep_start = datetime.fromtimestamp(int(session["startTimeMillis"]) / 1000)
                sleep_end = datetime.fromtimestamp(int(session["endTimeMillis"]) / 1000)
                sleep_duration = str(sleep_end - sleep_start)
                break  # Found a valid sleep session, no need to iterate further

# Convert sleep duration to string format
    if sleep_duration != "No sleep data available":
        sleep_duration = sleep_duration.split(".")[0]  # Remove milliseconds
    else:
        # No valid sleep session found
        sleep_duration = "No sleep data available"



    # Steps data request
    steps_data_request = {
        "aggregateBy": [{
            "dataTypeName": "com.google.step_count.delta",
            "dataSourceId": "derived:com.google.step_count.delta:com.google.android.gms:estimated_steps"
        }],
        "bucketByTime": { "durationMillis": 86400000 },
        "startTimeMillis": int(parse(seven_days_ago).timestamp() * 1000),
        "endTimeMillis": int(parse(now).timestamp() * 1000)
    }
    steps_data = fit_client.users().dataset().aggregate(userId='me', body=steps_data_request).execute()
    buckets = steps_data.get('bucket', [])
    steps = 'Unknown'
    if buckets:
        datasets = buckets[-1].get('dataset', [])
        if datasets:
            points = datasets[0].get('point', [])
            if points:
                values = points[0].get('value', [])
                if values:
                    steps = values[0].get('intVal', 'Unknown')
    else:
        steps = "No steps data available"

    # Heart rate data request
    heart_rate_data_request = {
        "aggregateBy": [{
            "dataTypeName": "com.google.heart_rate.bpm",
            "dataSourceId": "derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm"
        }],
        "bucketByTime": { "durationMillis": 86400000 },
        "startTimeMillis": int(parse(seven_days_ago).timestamp() * 1000),
        "endTimeMillis": int(parse(now).timestamp() * 1000)
    }
    heart_rate_data = fit_client.users().dataset().aggregate(userId='me', body=heart_rate_data_request).execute()
    buckets = heart_rate_data.get('bucket', [])
    heart_rate = 'Unknown'
    if buckets:
        datasets = buckets[-1].get('dataset', [])
        if datasets:
            points = datasets[0].get('point', [])
            if points:
                values = points[0].get('value', [])
                if values:
                    heart_rate = f"{round(values[0].get('fpVal', 'Unknown'))} bpm"
    else:
        heart_rate = "No heart rate data available"

    return {
        'name': name,
        'age': age,
        'weight': weight,
        'height': height,
        'sleep_duration': sleep_duration,
        'steps': steps,
        'heart_rate': heart_rate,
    }

@app.route('/')
def display_data():
    # Call the function to fetch Google Fit data
    data = get_google_data()

    # Print or log the data object
    print("Data object:", data)

    # Pass the data object to the HTML template and render it
    return render_template('template.html', data=data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/google_fit')
def google_fit():
    return render_template('google_fit_home.html')

@app.route('/disease_prediction')
def disease_prediction():
    global symptoms
    if symptoms is None:
        symptoms = X.columns.values
    return render_template('disease_form.html', symptoms=symptoms)

if __name__ == '__main__':
    app.run(debug=True)
