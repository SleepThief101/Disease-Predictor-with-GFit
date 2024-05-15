# Disease Prediction and Google Fit Data Integration

This project is a Flask-based web application that allows users to predict diseases based on their symptoms and integrates with the Google Fit API to fetch and display the user's health and fitness data.

## Features

- Disease prediction using Random Forest, Naive Bayes, and SVM models
- Symptom selection form for disease prediction
- Display of individual model predictions and a combined final prediction
- Accuracy scores and confusion matrices for each model
- Integration with the Google Fit API to fetch user data such as weight, height, sleep duration, steps, and heart rate
- Display of the user's Google Fit data on the web page
- Saving of prediction results to text files in the `output` directory
- Viewing of previous prediction results

## Prerequisites

- Python 3.x
- Google API credentials (OAuth 2.0 client ID and client secret)

## Installation

1. Clone the repository:
2. Install the required Python packages: pip install -r requirements.txt
3. Set up the Google API credentials:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a new project.
   - Enable the Google Fit API and People API for your project.
   - Create an OAuth 2.0 client ID and client secret for a desktop application.
   - Download the `credentials.json` file and place it in the project directory.

4. Run the Flask app: flask run
