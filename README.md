# 🏦 Bank Churn Predictor

An **AI-powered classification system** that predicts whether a bank customer is likely to **leave (churn)** or **stay** with the bank, based on key demographic, transactional, and behavioral data.
Built using **Deep Learning (Keras/TensorFlow)** and enhanced with **data preprocessing pipelines**, **feature encoding**, and **hyperparameter tuning**.

---

## 📊 Overview

Customer churn prediction helps financial institutions identify at-risk customers early and implement retention strategies.
This project uses a **supervised learning model** to predict churn from structured tabular data.

The model was trained on the **Bank Churn Modelling** dataset, and evaluated using multiple performance metrics (accuracy, precision, recall, F1-score, ROC-AUC).

---

## 🧠 Features

* Predicts whether a customer will **exit or stay** with the bank.
* Uses **Artificial Neural Networks (ANNs)** for robust prediction.
* Includes **feature scaling**, **label encoding**, and **one-hot encoding** pipelines.
* Supports **hyperparameter tuning** for optimal model performance.
* Easily deployable through a **Flask web app** (`app.py`).
* Pre-trained models available:

  * `model.h5` → Deep learning model
  * `regression_model.h5` → Logistic regression baseline
* Ready-to-use encoders and scaler saved as `.pkl` files.

---

## 🗂️ Repository Structure

```
BankChurnPredictor/
│
├── app.py                      # Flask web app for deployment
├── Churn_Modelling.csv         # Dataset used for training/testing
├── Prediction.ipynb            # Notebook for model prediction and testing
├── experiments.ipynb           # Model experiments and feature analysis
├── regression.ipynb            # Logistic regression baseline model
├── Hyperparameterfintunning.ipynb # Tuning parameters for ANN
│
├── model.h5                    # Trained ANN model
├── regression_model.h5         # Trained logistic regression model
│
├── label_encoder_gender.pkl    # Saved label encoder for Gender
├── one_hot_encoder_geo.pkl     # Saved one-hot encoder for Geography
├── scaler.pkl                  # Saved standard scaler for numerical features
│
├── requirements.txt            # Python dependencies
├── README.md                   # (You are here!)
└── LICENSE                     # MIT License
```

---

## 📚 Dataset Description

The dataset (`Churn_Modelling.csv`) contains details of 10,000 bank customers, including their:

| Feature             | Description                                              |
| ------------------- | -------------------------------------------------------- |
| **CustomerId**      | Unique identifier for each customer                      |
| **Surname**         | Customer surname                                         |
| **CreditScore**     | Customer credit rating                                   |
| **Geography**       | Country of the customer                                  |
| **Gender**          | Male/Female                                              |
| **Age**             | Customer’s age                                           |
| **Tenure**          | Number of years as a customer                            |
| **Balance**         | Account balance                                          |
| **NumOfProducts**   | Number of bank products used                             |
| **HasCrCard**       | Credit card ownership (1 = Yes, 0 = No)                  |
| **IsActiveMember**  | Account activity (1 = Active, 0 = Inactive)              |
| **EstimatedSalary** | Annual estimated income                                  |
| **Exited**          | Target variable (1 = Customer left, 0 = Customer stayed) |

---

## ⚙️ Data Preprocessing

Preprocessing steps performed before model training:

1. **Encoding categorical variables**

   * `Gender` → Label Encoding (`label_encoder_gender.pkl`)
   * `Geography` → One-Hot Encoding (`one_hot_encoder_geo.pkl`)

2. **Feature Scaling**

   * Standardized numeric features using `StandardScaler` (`scaler.pkl`)

3. **Splitting Data**

   * 80% Training, 20% Testing

4. **Feature Selection**

   * Removed `CustomerId`, `Surname`, and other non-contributive features.

---

## 🤖 Model Architecture

The **Artificial Neural Network (ANN)** model consists of:

| Layer          | Type  | Details                          |
| -------------- | ----- | -------------------------------- |
| Input Layer    | Dense | Input shape = number of features |
| Hidden Layer 1 | Dense | 6 neurons, ReLU activation       |
| Hidden Layer 2 | Dense | 6 neurons, ReLU activation       |
| Output Layer   | Dense | 1 neuron, Sigmoid activation     |

* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam
* **Metrics:** Accuracy
* **Epochs:** Tuned via experimentation (typically 100)

---

## 🔧 Hyperparameter Tuning

Conducted in `Hyperparameterfintunning.ipynb`:

* Batch size tuning (16, 32, 64)
* Learning rate optimization
* Dropout regularization
* Early stopping to prevent overfitting

---

## 📈 Evaluation Metrics

The model is evaluated using:

| Metric        | Description                                 |
| ------------- | ------------------------------------------- |
| **Accuracy**  | Overall correctness of the model            |
| **Precision** | Correct positive predictions                |
| **Recall**    | Ability to identify churned customers       |
| **F1 Score**  | Harmonic mean of precision and recall       |
| **ROC-AUC**   | Measures overall model discrimination power |

---

## 💻 Running the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/H-lamba/BankChurnPredictor.git
cd BankChurnPredictor
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask app

```bash
python app.py
```

Then open the browser at: **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

### 4️⃣ Predict churn for new data

You can enter customer information through the web interface to get **“Will Leave” / “Will Stay”** predictions.

---

## 🧪 Example Prediction

**Input:**

```
Geography: France
Gender: Female
Age: 42
Balance: 100000
Tenure: 8
NumOfProducts: 2
HasCrCard: 1
IsActiveMember: 1
CreditScore: 650
EstimatedSalary: 120000
```

**Output:**

```
Prediction: Customer will stay (Churn Probability: 0.18)
```

---

## 🧩 Requirements

The project requires the following Python packages:

```
numpy
pandas
matplotlib
scikit-learn
tensorflow
keras
flask
pickle-mixin
```

(Install automatically using `requirements.txt`.)

---

## 🚀 Future Improvements

* Implement **XGBoost** and **Random Forest** models for comparison.
* Add **SHAP/LIME** interpretability visualizations.
* Integrate **Streamlit UI** for a modern, interactive dashboard.
* Automate model retraining with new data.

---

## 🪪 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Himanshu Lamba**
🔗 [GitHub Profile](https://github.com/H-lamba)

---

## 🌟 Acknowledgements

* Dataset from Kaggle’s “Bank Churn Modelling” dataset.
* Inspired by real-world banking retention use cases.
* Built with ❤️ for learning, AI, and predictive analytics.

