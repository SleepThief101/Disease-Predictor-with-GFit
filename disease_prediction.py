from flask import Flask, render_template, request, redirect, url_for
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
from scipy.stats import mode
import os

app = Flask(__name__)

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
    return render_template('disease_form.html', symptoms=symptoms, rf_cm=None, nb_cm=None, svm_cm=None)

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

if __name__ == '__main__':
    app.run(debug=True)
