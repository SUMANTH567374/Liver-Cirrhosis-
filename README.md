# MLOps Project

# Liver Cirrhosis Prediction Project 🩺

This project aims to develop a robust machine learning pipeline for the **early prediction of liver cirrhosis** using clinical and laboratory data. It follows best practices in data science and MLOps, including data versioning, model tracking, experiment management, and reproducibility.

## 🔍 Objective

To predict liver cirrhosis status based on patient features, helping in early detection and intervention. The project includes:

- Data preprocessing and cleaning
- Feature selection using Random Forest
- Model training and hyperparameter tuning
- Evaluation using classification metrics and ROC-AUC
- Experiment tracking using **DVC**
- Model versioning and reproducibility

## 📦 Tech Stack

- **Python 3.12**
- **Scikit-learn**, **Imbalanced-learn**, **Pandas**, **Matplotlib**
- **DVC** for data and experiment tracking
- **Git** & **GitHub** for version control and collaboration
- **VS Code** for development

## 📁 Project Structure

Liver_Cirrhosis/
│
├── data/ # Raw, processed, final datasets
├── models/ # Trained ML models
├── reports/ # Metrics, JSON reports, evaluation results
├── graphs/ # Plots (confusion matrix, ROC curve, etc.)
├── logs/ # Logging files
├── src/ # Source code (data, features, models)
│ ├── data/ # Data ingestion and cleaning
│ ├── features/ # Feature selection
│ └── models/ # Training and evaluation
├── dvc.yaml # DVC pipeline stages
├── params.yaml # Configurable parameters
└── README.md # Project documentation



## ✅ Key Features

- Balanced dataset using **SMOTE** to handle class imbalance
- Top feature extraction using **Random Forest**
- Multiple model training and performance comparison
- Visualizations: confusion matrix, ROC curves
- DVC experiment tracking and reproducibility
- Remote DVC storage setup for shared pipelines

## 📊 Metrics Tracked

- **Accuracy**
- **Precision, Recall, F1-score**
- **ROC AUC Score**

## 🔁 Reproducibility

To reproduce the entire pipeline:

```bash
dvc repro

dvc exp run
dvc exp show


🌐 Remote Storage

Data and model artifacts are stored and versioned using a DVC remote set at:

bash
Copy
Edit
dvc remote add -d local_remote ./dvc_storage

🧠 Future Work

Integration of advanced models (XGBoost, LightGBM)

Deployment as an API using FastAPI

Real-time predictions from clinical input forms

Dockerizing the entire pipeline

