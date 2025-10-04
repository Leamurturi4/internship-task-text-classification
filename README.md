📰 News Category Classifier

A complete machine learning system for text classification.
The model classifies news articles into one of four categories: World, Sports, Business, or Sci/Tech.

📌 Project Overview

This project demonstrates an end-to-end machine learning workflow:

Data Loading & Analysis

Model Training & Comparison

Evaluation of Best Model

Deployment as CLI Tool & Web Application

🚀 Getting Started
1) Prerequisites

Python 3.8+

Recommended: a virtual environment (venv or conda)

2) Installation

Clone the repository and install the dependencies:

git clone <your-github-repo-link>
cd <your-project-folder>
pip install -r requirements.txt

3) Train the Model

Run the training script to download the dataset, train models, and save the best one:

python src/train_eval.py

4) Run Predictions

You can use the trained model in two ways.

A) Command-Line Interface (CLI):

# Classify a custom input
python src/predict.py "NASA's new telescope just discovered a distant galaxy"

# Or run with the default example
python src/predict.py


B) Web Application:

uvicorn src.serve:app --reload


Open your browser at: http://127.0.0.1:8000

📊 Dataset

Source: AG News Dataset (Hugging Face)

Link: https://huggingface.co/datasets/SetFit/ag_news

Size: 120k training samples, 7.6k test samples

Categories: 4 (World, Sports, Business, Sci/Tech)

🧠 Models & Approach

Preprocessing

Text converted to numerical features with TfidfVectorizer

Includes lowercasing, stop-word removal, and word-weighting

Models Trained

Multinomial Naive Bayes (baseline)

Logistic Regression (final selection)

Evaluation

Logistic Regression achieved the best performance

📈 Results

Final model performance on the test set:

              precision    recall  f1-score   support
World          0.930      0.902     0.916      1900
Sports         0.955      0.983     0.969      1900
Business       0.886      0.878     0.882      1900
Sci/Tech       0.887      0.896     0.892      1900

Accuracy: 91.5%

📂 Project Structure
.
├── models/              # Saved model files & evaluation plots
├── src/                 
│   ├── train_eval.py    # Train & evaluate models
│   ├── predict.py       # CLI prediction tool
│   └── serve.py         # FastAPI web application
├── templates/           
│   └── index.html       # Web app frontend
├── requirements.txt     # Dependencies
└── README.md            # Project documentation

⚙️ Technologies Used

Python (scikit-learn, FastAPI, Uvicorn)

Hugging Face Datasets

TfidfVectorizer

Logistic Regression / Naive Bayes
