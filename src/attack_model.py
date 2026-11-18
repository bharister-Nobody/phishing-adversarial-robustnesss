
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import textattack
from textattack.models.wrappers import ModelWrapper
from textattack.attack_recipes import DeepWordBugGao2018

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# --- Data Preprocessing Functions (from the original notebook) ---
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Custom ModelWrapper for TextAttack ---
class LSTMModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, max_len=150):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, text_list):
        sequences = self.tokenizer.texts_to_sequences(text_list)
        padded = pad_sequences(sequences, padding='post', maxlen=self.max_len)
        preds = self.model.predict(padded, verbose=0)
        # TextAttack expects probabilities for each class.
        # Since it's a binary classifier with sigmoid, we need to convert it.
        # Assuming 0 is 'Safe Email' and 1 is 'Phishing Email'
        # Output will be [[prob_safe, prob_phishing], ...]
        return np.array([[1-p[0], p[0]] for p in preds])

# --- Visualization Functions ---
def visualize_attack_results(log_file, successful_examples_file):
    df = pd.read_csv(log_file)

    # 1. Attack Success Rate (Matplotlib & Seaborn)
    success_rate = df["result_type"].value_counts(normalize=True).reset_index()
    success_rate.columns = ["Result Type", "Proportion"]

    plt.figure(figsize=(8, 6))
    sns.barplot(x="Result Type", y="Proportion", data=success_rate)
    plt.title("Attack Success Rate")
    plt.ylabel("Proportion")
    plt.xlabel("Result Type")
    plt.savefig("attack_success_rate.png")
    plt.close()
    print("Saved attack_success_rate.png")

    # 2. Distribution of Queries for Successful vs. Failed Attacks (Matplotlib & Seaborn)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="num_queries", hue="result_type", kde=True, bins=20)
    plt.title("Distribution of Queries for Successful vs. Failed Attacks")
    plt.xlabel("Number of Queries")
    plt.ylabel("Count")
    plt.savefig("num_queries_distribution.png")
    plt.close()
    print("Saved num_queries_distribution.png")

    # 3. Original Score vs. Perturbed Score for Successful Attacks (Plotly)
    successful_attacks = df[df["result_type"] == "Successful"].copy()
    if not successful_attacks.empty:
        # Convert scores to percentages for better visualization if they are probabilities
        successful_attacks["original_score_percent"] = successful_attacks["original_score"] * 100
        successful_attacks["perturbed_score_percent"] = successful_attacks["perturbed_score"] * 100

        fig = px.scatter(
            successful_attacks,
            x="original_score_percent",
            y="perturbed_score_percent",
            title="Original Score vs. Perturbed Score (Successful Attacks)",
            labels={
                "original_score_percent": "Original Score (%)",
                "perturbed_score_percent": "Perturbed Score (%)"
            },
            hover_data={
                "original_text": True,
                "perturbed_text": True,
                "original_score": ":.4f",
                "perturbed_score": ":.4f",
                "ground_truth_output": True,
                "original_output": True,
                "perturbed_output": True
            }
        )
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', name='Identity Line', line=dict(color='gray', dash='dash')))
        fig.write_html("scores_successful_attacks.html")
        print("Saved scores_successful_attacks.html")

        # Save successful adversarial examples
        successful_attacks.to_csv(successful_examples_file, index=False)
        print(f"Saved successful adversarial examples to {successful_examples_file}")

    else:
        print("No successful attacks to visualize scores or save examples.")

# --- Main Attack Logic ---
def run_attack(model_path, data_path):
    # Load data
    df = pd.read_csv(data_path)
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    df.dropna(inplace=True, axis=0)
    df.drop_duplicates(inplace=True)

    df["Email Text"] = df["Email Text"].apply(preprocess_text)

    # Encode labels
    le = LabelEncoder()
    df["Email Type"] = le.fit_transform(df["Email Type"])

    # Tokenizer and padding (must be the same as training)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['Email Text'])
    max_len = 150 # This was hardcoded in the notebook

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Create TextAttack model wrapper
    model_wrapper = LSTMModelWrapper(model, tokenizer, max_len)

    # Prepare dataset for TextAttack
    # TextAttack expects a list of (text, label) tuples
    dataset = textattack.datasets.Dataset(
        [(text, label) for text, label in zip(df['Email Text'], df['Email Type'])]
    )

    # --- Define Attack Recipe (Example: DeepWordBugGao2018) ---
    attack = DeepWordBugGao2018.build(model_wrapper)

    # --- Run Attack ---
    attack_args = textattack.AttackArgs(num_examples=10, log_to_csv="attack_log.csv", log_to_txt="attack_log.txt", disable_stdout=True)
    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()

    print("Attack completed. Results saved to attack_log.csv and attack_log.txt")

    # --- Visualize Results and Save Successful Examples ---
    visualize_attack_results("attack_log.csv", "successful_adversarial_examples.csv")

if __name__ == "__main__":
    model_file = r"C:\Users\USER\Ola_Adv_phishing\models\lstm_email_classifier.h5"
    data_file = r"C:\Users\USER\Ola_Adv_phishing\Data\raw\Phishing_Email.csv"
    run_attack(model_file, data_file)


