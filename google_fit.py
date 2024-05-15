from flask import Flask, render_template, session, redirect, url_for, request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from datetime import datetime, timedelta
from dateutil.parser import parse
import os
import json

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
                    weight = f"{values[0].get('fpVal', 'Unknown')} kg"
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
    sleep_dataset = fit_client.users().sessions().list(userId='me', fields='session', startTime=seven_days_ago, endTime=now).execute()
    sleep_duration = 'Unknown'
    if sleep_dataset.get('session'):
        for session in sleep_dataset['session']:
            if session["activityType"] == 72:
                sleep_start = datetime.fromtimestamp(int(session["startTimeMillis"]) / 1000)
                sleep_end = datetime.fromtimestamp(int(session["endTimeMillis"]) / 1000)
                sleep_duration = str(sleep_end - sleep_start)
    else:
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
def home():
    return render_template('google_fit_home.html')

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

if __name__ == '__main__':
    app.run(debug=True)