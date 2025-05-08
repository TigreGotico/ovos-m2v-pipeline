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


langs = ["en"]

datasets = []

base_models = ["minishlab/potion-base-2M",
               "minishlab/potion-base-4M",
               "minishlab/potion-base-8M",
               "minishlab/potion-base-32M",
               "minishlab/potion-retrieval-32M"]

# Metrics storage for later comparison
metrics_summary = []


csv_path = "merged_intents_dataset_en.csv"

# Load the dataset and remove duplicates based on the "utterance" column
df = pd.read_csv(csv_path)


# english model, use all loaded datasets for training
# Combine all datasets into a single training set
all_X = df["sentence"].values
all_y = df["label"].values

# Split the combined dataset into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.1, random_state=42)

# Initialize a classifier from one of the english models
for base_model in base_models:
    classifier = StaticModelForClassification.from_pretrained(model_name=base_model)

    # Train the english classifier on the combined data
    logger.info(f"Training english model {base_model}...")
    classifier.fit(X_train, y_train, max_epochs=25)

    # Predict using the trained english classifier
    logger.info(f"Predicting using the english model {base_model}...")
    y_pred = classifier.predict(X_test)

    # Evaluate the english model using various metrics
    logger.info(f"Evaluating english model {base_model}...")
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    logger.info(f"english Classification report:")
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"F1 Score: {f1}")
    logger.info("Confusion Matrix:")
    logger.info(f"\n{conf_matrix}")

    # Save the trained english model
    model_path = f"m2v_intents_{base_model.split('/')[-1]}"
    pipeline = classifier.to_pipeline()

    if os.path.isfile(model_path):
        os.remove(model_path)
    pipeline.save_pretrained(model_path)
    logger.info(f"english Model saved as {model_path}")

    # Save the evaluation metrics for the english model
    summary = {
        "language": "en",
        "model": base_model,
        "accuracy": accuracy,
        "f1_score": f1,
        "report": report
    }
    metrics_summary.append(summary)

    # Save the metrics summary to a markdown file
    with open(f"metrics_en_{base_model.split('/')[-1]}.md", "w") as f:
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
with open("model_comparison_en.md", "w") as f:
    f.write("# Model Comparison\n")
    f.write(comparison_df.to_markdown(index=False))
