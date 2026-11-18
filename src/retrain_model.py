
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re

# --- Data Preprocessing Functions (from the original notebook) ---
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Main Retraining Logic ---
def retrain_model(original_data_path, adversarial_data_path, original_model_path, new_model_path):
    # Load original data
    df_original = pd.read_csv(original_data_path)
    df_original.drop(["Unnamed: 0"], axis=1, inplace=True)
    df_original.dropna(inplace=True, axis=0)
    df_original.drop_duplicates(inplace=True)

    # Load adversarial data
    df_adversarial = pd.read_csv(adversarial_data_path)
    # Select only the perturbed text and its original label for retraining
    # Assuming 'ground_truth_output' is the correct label for the perturbed text
    df_adversarial_filtered = df_adversarial[['perturbed_text', 'ground_truth_output']].copy()
    df_adversarial_filtered.rename(columns={'perturbed_text': 'Email Text', 'ground_truth_output': 'Email Type'}, inplace=True)

    # Map integer labels back to original string labels
    # Assuming 0 is 'Phishing Email' and 1 is 'Safe Email' based on the original notebook's LabelEncoder behavior
    # It's crucial to verify this mapping from the original notebook or training process
    label_mapping = {0: 'Phishing Email', 1: 'Safe Email'}
    df_adversarial_filtered['Email Type'] = df_adversarial_filtered['Email Type'].map(label_mapping)

    # Combine datasets
    df_combined = pd.concat([df_original, df_adversarial_filtered], ignore_index=True)
    df_combined.dropna(inplace=True, axis=0)
    df_combined.drop_duplicates(inplace=True)

    df_combined["Email Text"] = df_combined["Email Text"].apply(preprocess_text)

    # Encode labels
    le = LabelEncoder()
    df_combined["Email Type"] = le.fit_transform(df_combined["Email Type"])

    # Prepare data for model training
    X = df_combined["Email Text"]
    y = df_combined["Email Type"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Tokenizer and padding (must be consistent with original model if possible, or retrain tokenizer)
    # For simplicity, we'll fit a new tokenizer on the combined data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    vocab_size = len(tokenizer.word_index) + 1
    max_len = 150 # This was hardcoded in the notebook

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_padded = pad_sequences(X_train_seq, padding='post', maxlen=max_len)
    X_test_padded = pad_sequences(X_test_seq, padding='post', maxlen=max_len)

    # Load the original model architecture (or create a new one if needed)
    # Assuming the original model is a Sequential model with an Embedding, LSTM, and Dense layer
    model = tf.keras.models.load_model(original_model_path)

    # Recompile the model (important for retraining)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Retrain the model
    print("Retraining model with adversarial examples...")
    history = model.fit(
        X_train_padded,
        y_train,
        epochs=5, # You might adjust this based on performance
        batch_size=32,
        validation_data=(X_test_padded, y_test),
        verbose=1
    )

    # Save the new model
    model.save(new_model_path)
    print(f"New robust model saved to {new_model_path}")

    # Save tokenizer for later use in evaluation
    import pickle
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Tokenizer saved to tokenizer.pickle")

if __name__ == "__main__":
    original_data_file = "/home/ubuntu/upload/Phishing_Email.csv"
    adversarial_data_file = "/home/ubuntu/successful_adversarial_examples.csv"
    original_model_file = "/home/ubuntu/upload/lstm_email_classifier.h5"
    new_model_file = "/home/ubuntu/lstm_email_classifier_robust.h5"
    retrain_model(original_data_file, adversarial_data_file, original_model_file, new_model_file)


