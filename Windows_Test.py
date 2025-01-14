from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import os
from dotenv import load_dotenv
import torch

# Carica le variabili dal file .env
load_dotenv("key.env")

# Path del modello
MODEL_PATH = os.getenv("MODEL_PATH_DEBERTA")

# Carica la configurazione del modello
config = AutoConfig.from_pretrained(MODEL_PATH)

# Verifica se le etichette sono presenti nella configurazione
if hasattr(config, 'id2label'):
    label_names = config.id2label
    n_labels = config.num_labels
    print("Numero di etichette:", n_labels)
    print("Etichette del modello:", label_names)
else:
    print("Le etichette non sono disponibili nella configurazione.")

device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

premise = "Emmanuel Macron is the President of USA"
hypothesis = "Emmanuel Macron is the President of France"

input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
output = model(input["input_ids"].to(device))  
prediction = torch.softmax(output["logits"][0], -1).tolist()
label_names = ["entailment", "not_entailment"]
prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
print(prediction)