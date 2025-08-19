from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Activation functions
def binary_step(x): return np.where(x >= 0, 1, 0)
def bipolar_step(x): return np.where(x >= 0, 1, -1)
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = np.clip(x, -500, 500)
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def relu(x): return np.maximum(0, x)

# Core training function
def train_model(X, y, activation, model_type, lr=0.01, epochs=100):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(epochs):
        for xi, target in zip(X, y):
            net_input = np.dot(xi, weights) + bias
            if activation == "binary":
                output = binary_step(net_input)
            elif activation == "bipolar":
                output = bipolar_step(net_input)
            elif activation == "sigmoid":
                output = sigmoid(net_input)
                output = 1 if output >= 0.5 else 0
            elif activation == "relu":
                output = relu(net_input)
                output = 1 if output >= 0.5 else 0
            elif activation == "softmax":
                output = softmax(np.array([net_input, 1 - net_input]).reshape(1, -1))
                output = np.argmax(output)

            error = target - (net_input if model_type == "adaline" else output)
            weights += lr * error * xi
            bias += lr * error

    return weights, bias

# Preprocessing
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    # Impute missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Standardize features
    features = df.iloc[:, :-1]
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(features)

    return df


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    file = request.files['dataset']
    model_type = request.form['model']
    lr = float(request.form['lr'])
    epochs = int(request.form['epochs'])

    df = pd.read_csv(file)
    df = preprocess(df)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    activations = ['binary', 'bipolar', 'sigmoid', 'relu', 'softmax']
    results = []

    for act in activations:
        weights, bias = train_model(X_train, y_train, activation=act, model_type=model_type, lr=lr, epochs=epochs)
        net_input = np.dot(X_test, weights) + bias
        if act == "binary":
            preds = binary_step(net_input)
        elif act == "bipolar":
            preds = bipolar_step(net_input)
            preds = np.where(preds == -1, 0, 1)
        elif act == "sigmoid":
            preds = sigmoid(net_input)
            preds = np.where(preds >= 0.5, 1, 0)
        elif act == "relu":
            preds = relu(net_input)
            preds = np.where(preds >= 0.5, 1, 0)
        elif act == "softmax":
            raw = np.stack([net_input, 1 - net_input], axis=1)
            preds = np.argmax(softmax(raw), axis=1)

        acc = accuracy_score(y_test, preds)
        results.append({
            'activation': act,
            'accuracy': round(acc * 100, 2),
            'weights': np.round(weights, 2).tolist(),
            'bias': round(bias, 2)
        })

    return render_template('results.html', results=results, model=model_type.upper())

if __name__ == '__main__':
    app.run(debug=True)
