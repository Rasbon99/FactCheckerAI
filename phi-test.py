from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')

if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Carica le variabili dal file .env
load_dotenv("key.env")

# Path del modello
MODEL_PATH = os.getenv("MODEL_PATH_PHI3")

# Carica il modello e il tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Configura la pipeline con il modello locale
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    framework="tf",
    device=gpus[0]
)

# Funzione di interazione
def chat_with_phi3():
    print("Benvenuto nel chat CLI con Phi-3.5! (Digita 'exit' per uscire)")
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
        outputs = generator(messages, max_new_tokens=128)
        response = outputs[0]["generated_text"]
        print(f"AI: {response}")

if __name__ == "__main__":
    chat_with_phi3()