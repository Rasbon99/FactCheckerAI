import os
import sys
import requests

from PIL import Image
import dotenv
import pandas as pd
import glob
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from log import Logger

class DashboardPipeline:
    def __init__(self, env_file="key.env"):
        
        dotenv.load_dotenv(env_file, override=True)
        self.logger = Logger(self.__class__.__name__).get_logger()
        
        self.logo = os.getenv('AI_IMAGE_UI')
        self.controller_url = os.getenv('CONTROLLER_SERVER_URL', 'http://127.0.0.1:8003')       
        
        # Inizializzazione del logger
        self.logger = Logger(self.__class__.__name__).get_logger()
        
        # Carica immagine nella sidebar
        self.image_sidebar = Image.open(self.logo).resize((300, 300))

        # Inizializza lo stato della sessione per i messaggi
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "To begin, state your claim..."}]

            
    def delete_chat_history(self):
        """
        Clears all tables from the database.
        """
        response = requests.post(f"{self.controller_url}/clean_conversations")
        if response.status_code == 200:
            st.sidebar.success("Chat history deleted successfully.")
        else:
            st.sidebar.success(f"Chat history elimination failed with status code {response.status_code}: {response.text}")
        
    
    def is_numeric_claim(self, claim_text):
        """
        Checks if the claim consists only of numbers.
        """
        return claim_text.isdigit()

    def get_response(self, claim):
        """
        Esegue una richiesta POST all'API /results, inviando il claim nel corpo della richiesta. 
        L'API ritorna un JSON contenente "status" e "response". Dal dizionario response vengono 
        estratti:
            - claim_title
            - claim_summary
            - query_result
            - sources: da ogni elemento della lista vengono presi solo "title" e "url"
            - graphs_folder: viene letta la cartella e vengono caricati tutti i file .jpg al suo interno,
                            convertendoli in immagini (PIL.Image)
                            
        Durante l'esecuzione viene mostrato uno spinner con Streamlit.
        
        Args:
            claim (Claim): L'oggetto claim da inviare all'API.
            
        Returns:
            tuple: (claim_title, claim_summary, query_result, sources, images)
        """
        with st.spinner("Processing claim..."):
            try:
                # Effettua la richiesta POST all'endpoint /results, inviando il claim nel body della richiesta
                response = requests.post(
                    f"{self.controller_url}/results",
                    json={"text": claim}  # Assicurati che il claim sia serializzabile in JSON
                )
                response.raise_for_status()
            except Exception as e:
                self.logger.error(f"Errore nella richiesta POST: {e}")
                st.error("Si è verificato un errore nell'elaborazione del claim.")
                return None
            
            data = response.json()
            
            result = data.get("response", {})

        # Estrazione dei dati dalla risposta
        claim_title   = result.get("claim_title", "")
        claim_summary = result.get("claim_summary", "")
        query_result  = result.get("query_result", "")
        
        # Per i sources, estrae solo "title" e "url" da ogni dizionario della lista
        sources = [
            {"title": src.get("title", ""), "url": src.get("url", "")}
            for src in result.get("sources", [])
        ]
        
        # Per graphs_folder: accede alla cartella e raccoglie tutte le immagini .jpg
        images = []
        graphs_folder = result.get("graphs_folder", "")
        if graphs_folder and os.path.isdir(graphs_folder):
            jpg_files = glob.glob(os.path.join(graphs_folder, "*.jpg"))
            for file in jpg_files:
                try:
                    img = Image.open(file)
                    images.append(img)
                except Exception as e:
                    self.logger.error(f"Errore nell'aprire l'immagine {file}: {e}")
        else:
            self.logger.warning("La cartella dei grafici non esiste o non è stata specificata.")
        
        return {'title': claim_title, 'summary' : claim_summary, 'response' : query_result, 'sources' : sources, 'images' : images}
    
    def display_claim_response(self, response):
        """
        Visualizza la risposta del sistema nella chat dell'assistente in Streamlit.

        Args:
            response (dict): Dizionario contenente i seguenti campi:
                - title (str): Il titolo del claim.
                - summary (str): Il riassunto del claim.
                - response (str): Il risultato della query.
                - sources (list): Lista di dizionari con chiavi "title" e "url".
                - images (list): Lista di immagini PIL da visualizzare.
        """
        # Creiamo il messaggio dell'assistente nella chat
        assistant_message = st.chat_message("assistant")

        # Stampiamo il titolo in grande
        assistant_message.markdown(f"<h2>{response['title'][2:]}</h2>", unsafe_allow_html=True)

        # Stampiamo il summary in trasparenza
        assistant_message.markdown(
            f"<p style='color: rgba(0, 0, 0, 0.6); font-size: 1.1em;'>{response['summary']}</p>", 
            unsafe_allow_html=True
        )

        # Stampiamo la query result in evidenza
        assistant_message.write(f"**{response['response']}**")

         # Area a tendina per visualizzare le fonti nella chat dell'assistente
        with assistant_message.expander("📌 Fonti"):
            for src in response.get('sources', []):
                st.markdown(f"- [{src['title']}]({src['url']})")

        # Visualizzazione delle immagini affiancate nella chat dell'assistente
        images = response.get('images', [])
        if images:
            with assistant_message.container():
                cols = st.columns(len(images))
                for col, img in zip(cols, images):
                    col.image(img)

    def get_conversations(self):
        with st.spinner("Processing conversations..."):
            try:
                # Effettua la richiesta GET all'endpoint /conversations
                response = requests.get(
                    f"{self.controller_url}/conversations")
                response.raise_for_status()
            except Exception as e:
                self.logger.error(f"Errore nella richiesta GET: {e}")
                st.error("Si è verificato un errore nell'elaborazione delle conversazioni.")
                return None
        
        data = response.json()

        return data.get("response", {})
    
    def display_images(self, images):
        # Visualizzazione delle immagini affiancate nella chat dell'assistente
        if images:
            cols = st.columns(len(images))
            for col, img in zip(cols, images):
                col.image(img)
       
    def run(self):
        with st.sidebar:
            st.title("Menu")
            option = st.radio("Choose an action:", ["New conversation", "View conversations"])
            st.image(self.image_sidebar)
            if st.button("Delete chat history"):
                self.delete_chat_history()
            
            if st.button("Exit Dashboard"):
                self.logger.info("Dashboard exited by user.")
                st.sidebar.warning("Exiting application...")
                os._exit(0)  # Termina il processo immediatamente
                

        if option == "View conversations":
            df_weekly = self.get_conversations()  # Recupera le conversazioni dal database
            
            df_weekly = pd.DataFrame(df_weekly)
            
            if not df_weekly.empty:
                st.sidebar.subheader("Past conversations:")
                
                # Crea una lista di opzioni per il selectbox: solo i claim
                conversation_list = df_weekly[['id', 'title']].sort_values(by='id', ascending=False)
                conversation_options = list(conversation_list['title'])
                
                selected_conversation = st.sidebar.selectbox("Select a conversation", conversation_options)
                
                # Estrai l'ID selezionato
                selected_id = conversation_list[conversation_list['title'] == selected_conversation]['id'].values[0]
                
                # Recupera la conversazione corrispondente
                conversation = df_weekly[df_weekly['id'] == selected_id].iloc[0]

                st.write(f"**❓ CLAIM**: {conversation['claim']}")

                # Mostra tutte le sources disponibili in una text_area
                if conversation['sources']:
                    sources_text = "\n\n".join(
                        [f"🔹 Title: [{src['title']}]({src['url']})" for src in conversation['sources']]
                    )
                else:
                    sources_text = "No sources available for this claim."

                st.write("**📚 SOURCES:**")
                
                st.markdown(sources_text)

                st.write("**🤖 RESPONSE**:")
                st.write(conversation['answer'])

                if conversation['images'] is not None:
                    self.display_images(conversation['images'])
            
            else:
                st.sidebar.info("⚠️ No conversations found.")
                st.info("⚠️ No conversations found. Start a new conversation to view it here.")


        else:
            # title "FOX AI" with fox emoji
            st.title("🦊 FOX AI")
            st.caption("🔍 Your personal assistant on fact-checking")
            
            if prompt := st.chat_input(max_chars=800):
                if self.is_numeric_claim(prompt):
                    st.chat_message("assistant").write(
                        "This claim is invalid because it consists only of numbers. The conversation will be deleted.")
                    st.session_state.messages = []
                    self.logger.warning("User entered an invalid numeric claim.")
                else:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.chat_message("user").write(prompt)

                    if len(st.session_state.messages) > 1:
                        claim = st.session_state.messages[-1]['content']
                        response = self.get_response(claim)
                        self.logger.debug(response)
                        self.display_claim_response(response)
                    else:
                        st.chat_message("assistant").write("Waiting for your claim...")

if __name__ == "__main__":
    dashboard = DashboardPipeline()
    dashboard.run()
