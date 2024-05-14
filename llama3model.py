import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
import torch

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from transformers import pipeline


import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Preprocess messages
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
data = pd.read_csv('ScamDataset', sep='\t', names=['Label', 'message'])

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(stemmed)

data['processed_message'] = data['message'].apply(preprocess)
texts = data['processed_message'].tolist()

data['Label'] = data['Label'].map({'normal': 0, 'fraud': 1})  # Adjust as necessary based on your actual labels
labels= data['Label'].tolist()


conversations = texts

# Load the text classification model
model_name = "meta-llama/Meta-Llama-3-8B"  # Adjust model name/path as necessary
classifier = pipeline("text-classification", model=model_name, device=-1)  # using GPU if available (device=0)

# Predict using the model in batches
def predict_in_batches(texts, batch_size=100):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = classifier(batch)
        results.extend(batch_results)
        print("Batch: " + str(i))
        print(batch_results)
    return results

# Run predictions
batch_size = 5  # Adjust batch size based on your computational resources
predictions = predict_in_batches(conversations, batch_size=batch_size)

# Extract labels from results and add to the dataframe
predicted_labels = [prediction['label'] for prediction in predictions]
data['prediction'] = predicted_labels

# Save the predictions back to a CSV
output_file_path = 'PredictedResults.csv'
data.to_csv(output_file_path, index=False)
print(f"Predictions completed and saved to {output_file_path}.")
