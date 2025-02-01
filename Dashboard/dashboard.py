import streamlit as st
import pandas as pd
import os
<<<<<<< Updated upstream
import sys
=======
>>>>>>> Stashed changes
from log import Logger
from datetime import datetime, timedelta
from PIL import Image

<<<<<<< Updated upstream
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Database.data_entities import Claim


# CONSTANTS
AI_IMAGE_UI=r"./Dashboard/AI_image.png"

=======
# CONST VARIABLES
AI_image_path = r'./Dashboard/AI_image.png'


>>>>>>> Stashed changes
class DashboardPipeline:
    def __init__(self):
        # Inizializzazione del logger
        self.logger = Logger("DashboardLogger", log_file="logger.log").get_logger()
        
        # Carica immagine nella sidebar
<<<<<<< Updated upstream
        self.image_sidebar = Image.open(AI_IMAGE_UI).resize((300, 300))
=======
        self.image_sidebar = Image.open(AI_image_path).resize((300, 300))
>>>>>>> Stashed changes

        # Inizializza lo stato della sessione per i messaggi
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "To begin, state your claim..."}]

        # Inizializza il logger solo una volta
        if "log_initialized" not in st.session_state:
            st.session_state["log_initialized"] = False

        if not st.session_state["log_initialized"]:
            self.logger.info("Dashboard initialized.")
            st.session_state["log_initialized"] = True
<<<<<<< Updated upstream
    
    def get_claim(self,claim):
        claim_obj=Claim(claim)

    def generate_response(self, claim):
        image_graph = AI_IMAGE_UI  # Percorso dell'immagine generata
=======

    def generate_response(self, claim):
        image_graph =   AI_image_path# Percorso dell'immagine generata
>>>>>>> Stashed changes
        self.logger.info(f"Generated response for claim: {claim}")
        return "HELLOHELLO", image_graph

    def save_chat_history(self, claim, answer, image_path):
        current_datetime = datetime.now()
        date = current_datetime.strftime('%Y-%m-%d')
        time = current_datetime.strftime('%H:%M:%S')

        if os.path.exists('chat_history.csv'):
            df = pd.read_csv('chat_history.csv')
            new_id = df['id'].max() + 1 if not df.empty else 1
        else:
            new_id = 1
            df = pd.DataFrame(columns=['id', 'date', 'time', 'claim', 'answer', 'image_path'])

        new_data = pd.DataFrame([[new_id, date, time, claim, answer, image_path]],
                                columns=['id', 'date', 'time', 'claim', 'answer', 'image_path'])
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv('chat_history.csv', index=False)

        self.logger.info(f"Saved chat history for claim: {claim}")

    def get_weekly_conversations(self):
        if os.path.exists('chat_history.csv'):
            df = pd.read_csv('chat_history.csv')
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            one_week_ago = datetime.now() - timedelta(weeks=1)
            self.logger.info("Retrieved weekly conversations.")
            return df[df['date'] >= one_week_ago]
        return pd.DataFrame()

    def delete_chat_history(self):
        if os.path.exists('chat_history.csv'):
            os.remove('chat_history.csv')
            self.logger.info("Chat history deleted.")
            st.sidebar.success("Chat history successfully deleted!")

    def is_numeric_claim(self, claim):
        return claim.isdigit()

    def run(self):
        with st.sidebar:
            st.title("Menu")
            option = st.radio("Choose an action:", ["New conversation", "View conversations"])
            st.image(self.image_sidebar)
            if st.button("Delete chat history"):
                self.delete_chat_history()

        if option == "View conversations":
            df_weekly = self.get_weekly_conversations()
            if not df_weekly.empty:
                st.sidebar.subheader("Past conversations:")
                conversation_list = df_weekly[['date', 'claim']].sort_values(by='date', ascending=False)
                conversation_list = conversation_list.apply(
                    lambda row: f"{row['date'].strftime('%Y-%m-%d')} - {row['claim']}", axis=1)
                selected_conversation = st.sidebar.selectbox("Select a conversation", conversation_list)
                selected_date, selected_claim = selected_conversation.split(" - ", 1)
                conversation = df_weekly[
                    (df_weekly['date'] == selected_date) & (df_weekly['claim'] == selected_claim)].iloc[0]

                st.write(f"**📅 DATE**: {conversation['date']}")
                st.write(f"**⏰ TIME**: {conversation['time']}")
                st.write(f"**❓ CLAIM**: {conversation['claim']}")
                st.write("**🤖 RESPONSE**:")
                st.text_area(value=conversation['answer'], label="none", label_visibility="collapsed", disabled=True)

                if os.path.exists(conversation['image_path']):
                    st.image(conversation['image_path'], caption="GraphRAG generated by LLM")
            else:
                st.write("No conversations recorded in the last week.")

        else:
            st.title("💬 Fact-checking AI")
            st.caption("🚀 Your personal assistant on fact-checking")

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
<<<<<<< Updated upstream
                        # Save claim to db
                        self.get_claim(claim)
=======
>>>>>>> Stashed changes
                        assistant_response, image_path = self.generate_response(claim)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                        st.chat_message("assistant").write(assistant_response)
                        st.chat_message("assistant").image(image_path)
                        self.save_chat_history(prompt, assistant_response, image_path)
                    else:
                        st.chat_message("assistant").write("Waiting for your claim...")


if __name__ == "__main__":
    dashboard = DashboardPipeline()
    dashboard.run()
