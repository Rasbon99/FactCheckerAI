from transformers import pipeline
import os
from dotenv import load_dotenv
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)

# Carica le variabili dal file .env
load_dotenv("key.env")

# Path del modello
MODEL_PATH = os.getenv("MODEL_PATH")

# Configura la pipeline con il modello locale
pipeline = pipeline(
    "text-generation",
    model=MODEL_PATH,
    device=gpus,
)

# Funzione di interazione
def chat_with_phi4():
    print("Benvenuto nel chat CLI con Phi-4! (Digita 'exit' per uscire)")
    system_message = {
        "role": "system",
        "content": "You are a medieval knight and must provide explanations to modern people."
    }
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Struttura i messaggi
        messages = [
            system_message,
            {"role": "user", "content": user_input}
        ]

        # Genera una risposta
        outputs = pipeline(messages, max_new_tokens=128)
        response = outputs[0]["generated_text"]
        print(f"AI: {response}")

if __name__ == "__main__":
    chat_with_phi4()