# Scam-detection-model
The project compares performance of different machine learning models to perform binary classification of phone call content to identify whether it is a "scam" or "normal" conversation. 

## Dataset
The dataset consist of one feature which is the conversation and the output with label either "fraud" or "normal". To help detect phone scam targeted at Singapore, conversations were generated with Generative AI tools that will add everday conversations in Singapore. All the data is combined on a single *ScamDataset.csv* file.

Source:
- https://www.kaggle.com/datasets/narayanyadav/fraud-call-india-dataset
- https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- https://github.com/wspr-ncsu/robocall-audio-dataset/blob/main/metadata.csv
- https://huggingface.co/datasets/alissonpadua/ham-spam-scam-toxic

Generated data:
![alt text](readme_images/image.png)

## Setup
### Prerequirsite
Python Jupyter Notebook was used to develop the code to evaluate the different model. This model can be run on a cloud environment service like Google CoLab or run on a local environment with a NVIDEA CUDA supported GPU.

1. Install the python liberies

``` 
    pip install -r requirements.txt
```

(For running on local environment, make sure NVIDEA's cuda toolkit and cuDNN library is installed if your machine has a NVIDEA GPU.)

2. For training and evaluating models (Naive bayes, SVM, Random Forest, Logistic Regression, LSTM)

Run conventional-models.ipynb

3. For training and evaluating models (Bert, RoBERTa and DistilBERT)

Run transformer_models.ipynb

4. For training and evaluating models (GPT2 and Llama 3)

- Run gpt2Model.ipynb
- Run llama3_Fine_Tuning.ipynb

# llama 3 verification
The use of llama 3 model would require to seek approval at Huggingface website. You are required to create an Huggingface account, fill out a form at https://huggingface.co/meta-llama/Meta-Llama-3-8B. (It should take about 1-2 working days). Next, generate an access token at the Huggingface website.

Profile -> Settings -> Access Tokens -> New Tokens

Next run the llama3_Fine_Tuning.ipynb code and when the code runs "!huggingface-cli login", it will prompt for the access token. Copy and paste the access token and press enter. 
(This step can be done on a seperate terminal. Run *huggingface-cli login*)

![alt text](readme_images/huggingface_login.png)


# Results

## Conventional Models
![NaivesBayes_matric](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/9c6eb675-2c2c-486a-a19b-11afa71371f2)
![SVM_Matric](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/ca4f3f31-4f1d-4db3-a7f5-0dccfdf12e05)
![RandomForest_Matric](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/5414a8ac-04ed-4cdc-bbd0-9ca35d28583c)
![LogisticRegression_Matric](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/75545410-4b0a-4fe9-953f-e0d2780aa14d)
![LSTM_matrics](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/fefe47a6-f4f6-4bf7-ae71-b58cbab5672b)
![Accuracy_Comparison_conventional](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/25a47bfa-14e1-442f-b749-32ed5d5c8a82)
![Precision_Comparison_conventional](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/04c802ba-4cfc-4c18-b267-3764eaf75b72)
![Recall_Comparison_conventional](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/6a666bc0-455f-4524-9bf1-70d647805cee)
![F1_Comparison_conventional](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/b61bc22c-cabc-4f21-9219-a512f5c42b36)

## Transformer Models
![Bert_evaluation](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/3f6ed292-3ae0-4963-81b5-3c082e3175c6)
![Bert_matric](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/df42d6dc-1527-4ce2-a342-a989432b1e09)
![Roberta_evaluation](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/4d067e3d-ae84-4020-9919-409a70b19e4f)
![Roberta_matric](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/908bce7b-d5fc-4c94-8134-6f7ae8e0890c)
![Distilbert_evaluation](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/889ae5c3-3ed4-4d52-a11a-6ed89b79eca2)
![Distilbert_matric](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/cdfdb8b5-ce01-4942-bbc6-8f03ff98783e)
![Accuracy_comparison_transformer](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/77d29485-b848-4962-96e4-93d3168d8c29)
![Precision_comparison_transformer](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/32ed0fdd-dd5a-41d5-b7e3-fac430eeedfa)
![Recall_comparison_transformer](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/636a151d-a01e-420a-8628-6484eed67f63)
![F1_comparison_transformer](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/d8643a39-e6da-448e-886f-6772ceea91e0)

## Large Language Model

### Before fine-tuned
#### GPT-2
![GPT_before_evaluation](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/01d89225-b422-4841-8ad0-0de8570f7bc9)
![GPT_before_matric](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/9d269ee0-3e52-4ed0-b3d6-6a7591de0881)
#### Llama-3
![llama_before_evaluation](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/6e9c15a6-5d9a-40f1-886b-90ad0d858480)
![llama_before_matric](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/16bb876a-cb86-42cd-8aae-0c7e20d4753f)

### After fine-tuned
#### GPT-2
![GPT_evaluation](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/c24c277c-3b36-4f28-874d-40234623f520)
![GPT_matric](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/3a982cf7-a6d1-4ab1-aff9-e430dda9b1d5)

#### Llama-3
![training_status](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/02007190-12d6-4116-8a62-562b53808cf9)
![llama_evaluation](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/25eb65cc-34d9-499c-802d-0af385ddfb76)
![llama_matric](https://github.com/PohYiJieNicholas/Scam-detection-model/assets/97501534/2365fbbe-66e4-4a29-b510-e32f73d5f1d4)


