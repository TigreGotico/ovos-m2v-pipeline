import logging
import os

import pandas as pd
import requests
from model2vec.train import StaticModelForClassification
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


def download_dataset(url, path):
    """Download the dataset if it does not already exist."""
    if not os.path.exists(path):
        logger.info(f"Downloading dataset from {url}")
        with open(path, "wb") as f:
            f.write(requests.get(url).content)


langs = ["en", "pt", "eu", "es", "gl", "nl", "fr", "de", "ca"]

datasets = []

base_models = ["minishlab/M2V_multilingual_output"]

# Metrics storage for later comparison
metrics_summary = []


csv_path = "merged_intents_dataset.csv"

# Load the dataset and remove duplicates based on the "utterance" column
df = pd.read_csv(csv_path)


# multilingual model, use all loaded datasets for training
# Combine all datasets into a single training set
all_X = df["sentence"].values
all_y = df["label"].values

# Split the combined dataset into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.1)

# Initialize a classifier from one of the multilingual models
for base_model in base_models:
    classifier = StaticModelForClassification.from_pretrained(model_name=base_model)

    # Train the multilingual classifier on the combined data
    logger.info(f"Training multilingual model {base_model}...")
    classifier.fit(X_train, y_train, max_epochs=25)

    # Predict using the trained multilingual classifier
    logger.info(f"Predicting using the multilingual model {base_model}...")
    y_pred = classifier.predict(X_test)

    # Evaluate the multilingual model using various metrics
    logger.info(f"Evaluating multilingual model {base_model}...")
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    logger.info(f"Multilingual Classification report:")
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"F1 Score: {f1}")
    logger.info("Confusion Matrix:")
    logger.info(f"\n{conf_matrix}")

    # Save the trained multilingual model
    multilingual_model_path = f"model_mul_{base_model.split('/')[-1]}"
    pipeline = classifier.to_pipeline()

    if os.path.isfile(multilingual_model_path):
        os.remove(multilingual_model_path)
    pipeline.save_pretrained(multilingual_model_path)
    logger.info(f"Multilingual Model saved as {multilingual_model_path}")

    # Save the evaluation metrics for the multilingual model
    summary = {
        "language": "multilingual",
        "model": base_model,
        "accuracy": accuracy,
        "f1_score": f1,
        "report": report
    }
    metrics_summary.append(summary)

    # Save the metrics summary to a markdown file
    with open(f"metrics_mul_{base_model.split('/')[-1]}.md", "w") as f:
        f.write("# Model Evaluation Metrics Summary\n")
        f.write(f"## {summary['language']} - Model: {summary['model']}\n")
        f.write(f"### Accuracy: {summary['accuracy']}\n")
        f.write(f"### F1 Score: {summary['f1_score']}\n")
        f.write(f"### Classification Report:\n```\n{summary['report']}\n```\n")
        f.write("\n")

# Save metrics to a DataFrame for better visualization and comparison
comparison_df = pd.DataFrame(columns=["Language", "Model", "Accuracy", "F1 Score"])

# Loop through the metrics summary to fill the comparison DataFrame
for summary in tqdm(metrics_summary, desc="Building Model Comparison", unit="model"):
    comparison_df = pd.concat([
        comparison_df,
        pd.DataFrame([{
            "Language": summary['language'],
            "Model": summary['model'],
            "Accuracy": summary['accuracy'],
            "F1 Score": summary['f1_score']
        }])
    ], ignore_index=True)

# Save the comparison DataFrame to a markdown table
with open("model_comparison.md", "w") as f:
    f.write("# Model Comparison\n")
    f.write(comparison_df.to_markdown(index=False))
