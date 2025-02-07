import os
import sys
import requests
from PIL import Image
import dotenv
import glob
import streamlit as st
from log import Logger

class DashboardPipeline:
    def __init__(self, env_file="key.env"):
        dotenv.load_dotenv(env_file, override=True)
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.logo = os.getenv('AI_IMAGE_UI')
        self.controller_url = os.getenv('CONTROLLER_SERVER_URL', 'http://127.0.0.1:8003')

        # Carica immagine nella sidebar
        self.image_sidebar = Image.open(self.logo).resize((100, 100))

        # Inizializzazione stato sessione
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Inizializza le variabili dello stato della sessione."""
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "To begin, state your claim..."}]
        if "view_mode" not in st.session_state:
            st.session_state.view_mode = "chat"  # Modalità di visualizzazione di default

    def _log_error(self, msg):
        """Gestione centralizzata degli errori."""
        self.logger.error(msg)
        st.error(msg)

    def delete_chat_history(self):
        """Clear chat history."""
        try:
            response = requests.post(f"{self.controller_url}/clean_conversations")
            response.raise_for_status()
            st.sidebar.success("Chat history deleted successfully.")
        except requests.exceptions.RequestException as e:
            self._log_error(f"Error deleting chat history: {e}")

    def is_numeric_claim(self, claim_text):
        """Checks if the claim consists only of numbers."""
        return claim_text.isdigit()

    def get_response(self, claim):
        """Request response from controller."""
        with st.spinner("Processing claim..."):
            try:
                response = requests.post(
                    f"{self.controller_url}/results",
                    json={"text": claim}
                )
                response.raise_for_status()
                data = response.json()
                result = data.get("response", {})

                claim_title = result.get("claim_title", "")
                claim_summary = result.get("claim_summary", "")
                query_result = result.get("query_result", "")
                sources = [{"title": src.get("title", ""), "url": src.get("url", "")} for src in result.get("sources", [])]

                images = self._load_images_from_folder(result.get("graphs_folder", ""))
                return {'title': claim_title, 'summary': claim_summary, 'response': query_result, 'sources': sources, 'images': images}
            except requests.exceptions.RequestException as e:
                self._log_error(f"Error in POST request: {e}")
                return None

    def _load_images_from_folder(self, folder):
        """Carica immagini dalla cartella specificata."""
        images = []
        if folder and os.path.isdir(folder):
            for file in glob.glob(os.path.join(folder, "*.jpg")):
                try:
                    img = Image.open(file)
                    images.append(img)
                except Exception as e:
                    self.logger.error(f"Error opening image {file}: {e}")
        else:
            self.logger.warning(f"Graphs folder does not exist or was not specified: {folder}")
        return images

    def display_message(self, role, message, avatar="🦊"):
        """Gestisce la visualizzazione dei messaggi di chat."""
        chat_msg = st.chat_message(role, avatar=avatar)
        chat_msg.write(message)

    def display_claim_response(self, response):
        """
        Displays the system's response in the chat interface in Streamlit.
        """
        assistant_message = st.chat_message("assistant", avatar="🦊")

        # Styling per il titolo
        title_html = f"<h2 style='color: rgba(0, 0, 0, 0.9); font-size: 1.5em; line-height: 1.4;'>{response['title'][2:]}</h2>"
        assistant_message.markdown(title_html, unsafe_allow_html=True)

        # Styling per il sommario
        summary_html = f"<p style='color: rgba(0, 0, 0, 0.6); font-size: 1.1em; line-height: 1.6;'>{response['summary']}</p>"
        assistant_message.markdown(summary_html, unsafe_allow_html=True)

        # Risposta principale
        assistant_message.write(f"{response['response']}")

        # Se ci sono fonti, mostra nell'expander
        with assistant_message.expander("📌 Sources"):
            for src in response.get('sources', []):
                st.markdown(f"- [{src['title']}]({src['url']})")

        # Gestione delle immagini
        images = response.get('images', [])
        if images:
            with assistant_message.container():
                cols = st.columns(len(images))
                for col, img in zip(cols, images):
                    col.image(img)

    def get_conversations(self):
        """Recupera le conversazioni."""
        with st.spinner("Processing conversations..."):
            try:
                response = requests.get(f"{self.controller_url}/conversations")
                response.raise_for_status()
                data = response.json()
                return data.get("response", [])
            except requests.exceptions.RequestException as e:
                self._log_error(f"Error in GET request: {e}")
                return []

    def display_conversation(self, conversation):
        """Visualizza una conversazione storica in un'unica risposta."""
        # Funzione centralizzata per visualizzare i messaggi
        def display_message(role, content, avatar=None):
            if role == "assistant":
                message = st.chat_message("assistant", avatar=avatar)
            else:
                message = st.chat_message("user")
            message.markdown(content, unsafe_allow_html=True)

        # Concatenazione di titolo, claim e risposta in un'unica stringa
        content = f"""
        <h2 style='color: rgba(0, 0, 0, 0.9); font-size: 1.5em; line-height: 1.4;'>{conversation['title']}</h2>
        """
        
        if conversation['claim']:
            content += f"<p style='color: rgba(0, 0, 0, 0.6); font-size: 1.1em; line-height: 1.6;'>{conversation['claim']}</p>"

        content += f"<p style='font-size: 1.1em; line-height: 1.6;'>{conversation['answer']}</p>"

        # Visualizzazione unica della conversazione
        display_message("assistant", content, avatar="🦊")

        # Visualizzazione delle fonti
        with st.expander("📌 Sources"):
            if conversation['sources']:
                sources_text = "\n\n".join([f"- [{src['title']}]({src['url']})" for src in conversation['sources']])
                st.markdown(sources_text)
            else:
                st.markdown("No sources available for this claim.")

        # Visualizzazione delle immagini
        if conversation.get('images'):
            cols = st.columns(len(conversation['images']))
            for col, img in zip(cols, conversation['images']):
                col.image(img)


    def get_conversation_by_id(self, convo_id):
        """Recupera una conversazione tramite ID."""
        conversations = self.get_conversations()
        return next((convo for convo in conversations if convo['id'] == convo_id), None)

    def run(self):
        """Funzione principale per avviare il dashboard."""
        st.title("🦊 FOX AI")
        st.caption("🔍 Your personal assistant on fact-checking")

        if st.session_state.view_mode == "chat":
            prompt = st.chat_input()
            if prompt:
                if self.is_numeric_claim(prompt):
                    self.display_message("assistant", "This claim is invalid because it consists only of numbers.")
                    st.session_state.messages = []
                else:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    self.display_message("user", prompt)

                    if len(st.session_state.messages) > 1:
                        claim = st.session_state.messages[-1]['content']
                        response = self.get_response(claim)
                        self.display_claim_response(response)

        else:
            if 'selected_conversation' in st.session_state:
                selected_conversation = st.session_state.selected_conversation
                self.display_conversation(selected_conversation)
            else:
                self.display_message("assistant", "Waiting for your claim...")

        with st.sidebar:
            if st.sidebar.button("➕ New Conversation"):
                st.session_state.view_mode = "chat"
                st.session_state.messages = []
                st.session_state.selected_conversation = None
                st.rerun()

            if st.sidebar.button("🗑️ Delete Chat History"):
                self.delete_chat_history()

            if st.button("Exit Dashboard"):
                st.stop()

            st.image(self.image_sidebar)

            st.sidebar.title("Chat History")
            st.sidebar.text_input("Search claims...", key="search_query", placeholder="Type to search")

            conversations = self.get_conversations()
            filtered_conversations = [convo for convo in conversations if st.session_state.search_query.lower() in convo.get("title", "").lower()]
            filtered_conversations = filtered_conversations[::-1]  # Ordina le conversazioni per data

            for i, convo in enumerate(filtered_conversations):
                if st.sidebar.button(convo['title'][:50] + ("..." if len(convo['title']) > 50 else ""), key=f"convo_{i}"):
                    st.session_state.view_mode = "history"
                    st.session_state.selected_conversation = convo
                    st.rerun()

if __name__ == "__main__":
    dashboard = DashboardPipeline()
    dashboard.run()
