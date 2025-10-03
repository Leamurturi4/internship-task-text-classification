import joblib, numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
from collections import Counter

print("Loading AG News dataset...")
ds = load_dataset("ag_news")
X_train = [x["text"] for x in ds["train"]]
y_train = [x["label"] for x in ds["train"]]
X_test  = [x["text"] for x in ds["test"]]
y_test  = [x["label"] for x in ds["test"]]
print("Dataset loaded.")

print("\n--- Exploratory Analysis ---")
LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

print(f"Total training samples: {len(X_train)}")
print(f"Total test samples:     {len(X_test)}")

train_counts = Counter(y_train)
test_counts = Counter(y_test)

print("\nTraining data distribution:")
for label_id, count in sorted(train_counts.items()):
    print(f"- Class '{LABELS[label_id]}' (ID: {label_id}): {count} samples")

print("\nTest data distribution:")
for label_id, count in sorted(test_counts.items()):
    print(f"- Class '{LABELS[label_id]}' (ID: {label_id}): {count} samples")
print("----------------------------\n")

print("Building model pipelines...")
pipe_nb  = make_pipeline(
    TfidfVectorizer(stop_words="english"),
    MultinomialNB()
)
pipe_log = make_pipeline(
    TfidfVectorizer(stop_words="english"),
    LogisticRegression(max_iter=1000)
)
print("Pipelines built.")

print("Training Naive Bayes model...")
pipe_nb.fit(X_train, y_train)
print("Training Logistic Regression model...")
pipe_log.fit(X_train, y_train)
print("Training complete.")

def eval_model(name, pipe):
    print(f"\nEvaluating model: {name}...")
    preds = pipe.predict(X_test)
    print(f"\n=== Classification Report: {name} ===")
    print(classification_report(y_test, preds, target_names=LABELS.values(), digits=3))
    cm = confusion_matrix(y_test, preds)
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.colorbar();
    tick_marks = np.arange(len(LABELS))
    plt.xticks(tick_marks, LABELS.values(), rotation=45)
    plt.yticks(tick_marks, LABELS.values())
    plt.tight_layout()
    
    os.makedirs("models", exist_ok=True)
    plot_path = f"models/confusion_{name}.png"
    plt.savefig(plot_path)
    print(f"Confusion matrix saved to {plot_path}")

eval_model("MultinomialNB", pipe_nb)
eval_model("LogReg", pipe_log)

print("\nComparing model accuracy...")
acc_nb  = pipe_nb.score(X_test, y_test)
acc_log = pipe_log.score(X_test, y_test)
best = ("LogReg", pipe_log) if acc_log >= acc_nb else ("MultinomialNB", pipe_nb)
best_name, best_pipe = best
print(f"Best model: {best_name} (accuracy = {best_pipe.score(X_test, y_test):.4f})")

model_path = "models/model.joblib"
joblib.dump(best_pipe, model_path)
print(f"Best model pipeline saved to -> {model_path}")