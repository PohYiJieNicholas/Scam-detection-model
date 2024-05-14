import pandas as pd
from transformers import pipeline

# Step 1: Load your dataset
file_path = 'ScamDataset'  # Adjust the path and filename as needed
data = pd.read_csv('ScamDataset', sep='\t', names=['Label', 'message'])

conversations = data['message']  # Adjust this column name based on your dataset

# Step 2: Load the Meta-Llama model
model_name = "meta-llama/Meta-Llama-3-8B"
classifier = pipeline("text-classification", model=model_name)

# Step 3: Predict the category of each conversation
results = []
i = 0
for conversation in conversations:
    result = classifier(conversation)
    results.append(result[0]['label'])  # Assumes that the model outputs labels directly relevant to your categories
    print(str(i) + ": " + results[i])
    i+=1

# Step 4: Add predictions back to the dataframe
data['prediction'] = results

# Step 5: Output the results
print(data[['conversation_text', 'prediction']])
data.to_csv('PredictedResults.csv', index=False)  # Save the results to a new CSV file

print("Predictions completed and saved to PredictedResults.csv.")
