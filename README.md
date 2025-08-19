#  Neural Network Activation Function Simulator

A Flask-based web application that implements and compares different activation functions in neural networks using Perceptron and ADALINE models.

##  Features

### **Activation Functions Implemented**
- **Binary Step Function**: Classic threshold-based activation
- **Bipolar Step Function**: Returns +1 or -1 based on input
- **Sigmoid Function**: Smooth S-shaped curve with gradient
- **ReLU (Rectified Linear Unit)**: Linear for positive inputs, zero for negative
- **Softmax Function**: Probability distribution over multiple classes

### **Model Types**
- **Perceptron**: Basic linear classifier with binary output
- **ADALINE**: Adaptive linear neuron with continuous output during training

### **Smart Preprocessing**
- **Missing Value Imputation**: Automatic handling of null values
- **Label Encoding**: Converts categorical variables to numerical format
- **Feature Standardization**: Normalizes data using StandardScaler
- **Train-Test Split**: 80-20 split for model validation

##  How It Works

1. **Upload Dataset**: CSV file with features and target variable (last column)
2. **Configure Parameters**: Choose model type, learning rate, and epochs
3. **Train & Compare**: All activation functions are trained simultaneously
4. **View Results**: Compare accuracy, weights, and bias for each activation function

##  Project Structure

```
├── app.py                 # Main Flask application
├── templates/
│   ├── index.html        # Upload and configuration page
│   └── results.html      # Results comparison page
├── static/
    └── style.css         # Custom styling for UI

```

##  Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd neural-network-simulator
   ```

2. **Install dependencies**
   ```bash
   pip install flask pandas numpy scikit-learn
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open browser** and navigate to `http://localhost:5000`

##  Input Requirements

- **Dataset Format**: CSV file
- **Structure**: Features in all columns except the last one (target variable)
- **Target Variable**: Binary classification (0/1 or categorical)
- **File Size**: Reasonable size for web upload

##  Configuration Options

- **Model Type**: Perceptron or ADALINE
- **Learning Rate**: Typically between 0.001 and 0.1
- **Epochs**: Number of training iterations (50-1000 recommended)

##  Output

For each activation function, the simulator provides:
- **Accuracy Score**: Performance on test data
- **Learned Weights**: Feature importance values
- **Bias Term**: Model intercept value

##  Technical Implementation

- **Backend**: Flask web framework
- **Frontend**: Custom HTML templates with CSS styling
- **ML Library**: scikit-learn for preprocessing and evaluation
- **Numerical Computing**: NumPy for mathematical operations
- **Data Handling**: Pandas for CSV processing
- **UI/UX**: Responsive HTML forms with custom CSS styling
- **Gradient Clipping**: Prevents numerical overflow in sigmoid/softmax

##  Use Cases

- **Educational**: Learn how different activation functions behave
- **Research**: Compare activation function performance on custom datasets
- **Prototyping**: Quick testing of neural network concepts
- **Analysis**: Understand weight distributions across different activations

##  Notes

- Designed for **binary classification** problems
- Automatic preprocessing handles most data quality issues
- Results may vary based on dataset characteristics and hyperparameters
- For production use, consider additional validation techniques

##  Dependencies

```
Flask>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

---

*Perfect for students, researchers, and ML enthusiasts exploring neural network fundamentals!*
