import os
import sys
import time
import io

from PIL import Image
import dotenv
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Database.data_entities import Claim
from Database.sqldb import Database
from log import Logger

dotenv.load_dotenv(dotenv_path='key.env')

class DashboardPipeline:
    def __init__(self, env_file="key.env"):
        
        dotenv.load_dotenv(env_file, override=True)
        
        self.db_path = os.getenv('SQLITE_DB_PATH')
        self.logo = os.getenv('AI_IMAGE_UI')
        
        # Inizializzazione del database
        self.db = Database(self.db_path)
        
        # Inizializzazione del logger
        self.logger = Logger("DashboardLogger").get_logger()
        
        # Carica immagine nella sidebar
        self.image_sidebar = Image.open(self.logo).resize((300, 300))

        # Inizializza lo stato della sessione per i messaggi
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "To begin, state your claim..."}]

        # Inizializza il logger solo una volta
        if "log_initialized" not in st.session_state:
            st.session_state["log_initialized"] = False

        if not st.session_state["log_initialized"]:
            self.logger.info("Dashboard initialized.")
            st.session_state["log_initialized"] = True
            
    def delete_chat_history(self):
        """
        Clears all tables from the database.
        """
        self.db.delete_all_conversations()
        st.sidebar.success("Chat history deleted successfully.")
        
    def create_claim(self, claim_text):
        """
        Creates a Claim object from the given claim text.
        
        Args:
            claim_text (str): The claim text.
        
        Returns:
            Claim: The Claim object.
        """
        claim = Claim(claim_text)
        return claim
    
    def is_numeric_claim(self, claim_text):
        """
        Checks if the claim consists only of numbers.
        """
        return claim_text.isdigit()

    def get_response(self, claim):
        """
        Retrieves the text and Image from the answer associated with the given claim.
        
        Args:
            claim (Claim): The claim object.
            
        Returns:
            answer_text (str): The text of the answer.
            image (Image): The image associated with the answer.
        """
        chat_message_shown = False
        
        while claim.has_answer() is False:
            if not chat_message_shown:
                st.chat_message("assistant").write("I am still processing your claim. Please wait a moment...")
                
                self.logger.info("Processing claim...")
                chat_message_shown = True
            with st.spinner("Processing claim..."):
                time.sleep(int(os.getenv('CLAIM_PROCESSING_TIME')))    
        answer, image = claim.get_answer()
        
        return answer, image
    
    def get_conversations(self):
        """
        Retrieves the conversations from the database, including associated sources.

        Returns:
            pd.DataFrame: A DataFrame containing the conversations and their sources.
        """
        
        # Query per ottenere le conversazioni con risposta e immagine
        query = """
        SELECT c.id, c.text, a.answer, a.image 
        FROM claims c
        INNER JOIN answers a ON c.id = a.claim_id
        """
        
        rows = self.db.fetch_all(query)

        if not rows:
            return pd.DataFrame(columns=["id", "claim", "answer", "image", "sources"])

        conversations = []
        
        for row in rows:
            claim_id = row[0]
            
            # Query per ottenere le sources associate al claim
            sources_query = """
            SELECT title, url, body 
            FROM sources 
            WHERE claim_id = ?
            """
            sources_rows = self.db.fetch_all(sources_query, (claim_id,))

            # Formattare le sources come una lista di dizionari
            sources = [{"title": s[0], "url": s[1], "body": s[2]} for s in sources_rows]

            conversations.append({
                "id": claim_id,
                "claim": row[1],
                "answer": row[2],
                "image": Image.open(io.BytesIO(row[3])) if row[3] else None,
                "sources": sources  # Aggiunta delle sources
            })

        return pd.DataFrame(conversations)
            
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
            
            if not df_weekly.empty:
                st.sidebar.subheader("Past conversations:")
                
                # Crea una lista di opzioni per il selectbox: solo i claim
                conversation_list = df_weekly[['id', 'claim']].sort_values(by='id', ascending=False)
                conversation_options = list(conversation_list['claim'])
                
                selected_conversation = st.sidebar.selectbox("Select a conversation", conversation_options)
                
                # Estrai l'ID selezionato
                selected_id = conversation_list[conversation_list['claim'] == selected_conversation]['id'].values[0]
                
                # Recupera la conversazione corrispondente
                conversation = df_weekly[df_weekly['id'] == selected_id].iloc[0]

                st.write(f"**❓ CLAIM**: {conversation['claim']}")

                # Mostra tutte le sources disponibili in una text_area
                if conversation['sources']:
                    sources_text = "\n\n".join(
                        [f"🔹 Title: {src['title']}\n🔗 URL: {src['url']}\n📄 Body: {src['body']}" for src in conversation['sources']]
                    )
                else:
                    sources_text = "No sources available for this claim."

                st.write("**📚 SOURCES:**")
                
                st.text_area(label="none", label_visibility="collapsed", value=sources_text, height=200, disabled=True)

                st.write("**🤖 RESPONSE**:")
                st.text_area(value=conversation['answer'], label="none", label_visibility="collapsed", disabled=True)

                if conversation['image'] is not None:
                    st.image(conversation['image'], caption="GraphRAG generated by LLM")
            
            else:
                st.sidebar.info("⚠️ No conversations found.")
                st.info("⚠️ No conversations found. Start a new conversation to view it here.")


        else:
            # title "FOX AI" with fox emoji
            st.title("🦊 FOX AI")
            st.caption("🔍 Your personal assistant on fact-checking")
            
            if prompt := st.chat_input():
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
                        # Save claim to db
                        claim = self.create_claim(claim)
                        assistant_response, image = self.get_response(claim)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                        st.chat_message("assistant").write(assistant_response)
                        st.chat_message("assistant").image(image)
                    else:
                        st.chat_message("assistant").write("Waiting for your claim...")

if __name__ == "__main__":
    dashboard = DashboardPipeline()
    dashboard.run()
