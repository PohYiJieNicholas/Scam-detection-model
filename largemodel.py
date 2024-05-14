import pandas as pd
import numpy as np
import torch 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Preprocess messages
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('ScamDataset', sep='\t', names=['Label', 'message'])

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(stemmed)

df['processed_message'] = df['message'].apply(preprocess)
X = df['processed_message']

df['Label'] = df['Label'].map({'normal': 0, 'fraud': 1})  # Adjust as necessary based on your actual labels
y= df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate a model
def evaluate_model(model_name, X_train, X_test, y_train, y_test):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_name)


    # Tokenization
    # Tokenization and conversion to tensors
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")

    train_encodings = {key: val.to(model.device) for key, val in train_encodings.items()}
    test_encodings = {key: val.to(model.device) for key, val in test_encodings.items()}

    # Prediction
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Turn off gradients for prediction, saves memory and computations
        train_preds = model(**train_encodings)
        test_preds = model(**test_encodings)
    # train_preds = model(**{k: v.to(model.device) for k, v in train_encodings.items()})
    # test_preds = model(**{k: v.to(model.device) for k, v in test_encodings.items()})
    
    # Convert logits to labels
    train_labels = train_preds.logits.argmax(axis=1).cpu().numpy()
    test_labels = test_preds.logits.argmax(axis=1).cpu().numpy()
    
    # Accuracy
    train_accuracy = accuracy_score(y_train, train_labels)
    test_accuracy = accuracy_score(y_test, test_labels)
    
    return train_accuracy, test_accuracy



# Evaluate models
gpt_accuracy = evaluate_model('gpt2', X_train, X_test, y_train, y_test)
llama_accuracy = evaluate_model('facebook/llama-2', X_train, X_test, y_train, y_test)

print(f"GPT Model Accuracy: {gpt_accuracy}")
print(f"LLaMA Model Accuracy: {llama_accuracy}")