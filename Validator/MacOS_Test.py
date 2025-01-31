
from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification, AutoConfig
import os
from dotenv import load_dotenv
import tensorflow as tf

# Controllo delle GPU disponibili
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU configurata con memory growth.")
    except RuntimeError as e:
        print(e)

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
    
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Premessa e ipotesi
premise = "Emmanuel Macron is the President of USA"
hypothesis = "Emmanuel Macron is the President of France"

# Tokenizzazione
inputs = tokenizer(premise, hypothesis, truncation=True, return_tensors="tf")

# Inferenza
outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

# Softmax e interpretazione
predictions = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
label_names = ["entailment", "not_entailment"]
prediction = {name: round(pred * 100, 1) for pred, name in zip(predictions, label_names)}
print(prediction)