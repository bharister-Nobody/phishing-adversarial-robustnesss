
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
import pickle

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
        return np.array([[1-p[0], p[0]] for p in preds])

# --- Main Evaluation Logic ---
def evaluate_model_robustness(model_path, tokenizer_path, data_path, log_file_prefix):
    # Load data
    df = pd.read_csv(data_path)
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    df.dropna(inplace=True, axis=0)
    df.drop_duplicates(inplace=True)

    df["Email Text"] = df["Email Text"].apply(preprocess_text)

    # Encode labels
    le = LabelEncoder()
    df["Email Type"] = le.fit_transform(df["Email Type"])

    # Load the trained model and tokenizer
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Create TextAttack model wrapper
    model_wrapper = LSTMModelWrapper(model, tokenizer)

    # Prepare dataset for TextAttack
    dataset = textattack.datasets.Dataset(
        [(text, label) for text, label in zip(df['Email Text'], df['Email Type'])]
    )

    # --- Define Attack Recipe (DeepWordBugGao2018) ---
    attack = DeepWordBugGao2018.build(model_wrapper)

    # --- Run Attack ---
    attack_args = textattack.AttackArgs(
        num_examples=10, 
        log_to_csv=f"{log_file_prefix}_log.csv", 
        log_to_txt=f"{log_file_prefix}_log.txt", 
        disable_stdout=True
    )
    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()

    print(f"Attack completed for {model_path}. Results saved to {log_file_prefix}_log.csv and {log_file_prefix}_log.txt")

if __name__ == "__main__":
    original_model_file = "/home/ubuntu/upload/lstm_email_classifier.h5"
    robust_model_file = "/home/ubuntu/lstm_email_classifier_robust.h5"
    data_file = "/home/ubuntu/upload/Phishing_Email.csv"
    tokenizer_file = "/home/ubuntu/tokenizer.pickle"

    print("Evaluating original model...")
    evaluate_model_robustness(original_model_file, tokenizer_file, data_file, "original_model_attack")

    print("\nEvaluating robust model...")
    evaluate_model_robustness(robust_model_file, tokenizer_file, data_file, "robust_model_attack")


