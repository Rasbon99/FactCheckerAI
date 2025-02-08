import os
import sys
import requests

from PIL import Image
import dotenv
import glob
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from log import Logger

class DashboardPipeline:
    def __init__(self, env_file="key.env"):
        """
        Initializes the dashboard pipeline by loading environment variables 
        and setting up the logger and session state.
        
        Args:
            env_file (str): The path to the environment file containing necessary credentials and configurations.
        
        Raises:
            Exception: If environment variables cannot be loaded or other initialization errors occur.
        """
        dotenv.load_dotenv(env_file, override=True)
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.logo = os.getenv('AI_IMAGE_UI')
        self.controller_url = os.getenv('CONTROLLER_API_URL', 'http://127.0.0.1:8003')

        # Load image into the sidebar
        self.image_sidebar = Image.open(self.logo).resize((100, 100))

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """
        Initializes session state variables for managing the chat messages 
        and view mode (chat or history) in the Streamlit app.
        
        Raises:
            Exception: If there is an error during session state initialization.
        """
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "To begin, state your claim..."}]
        if "view_mode" not in st.session_state:
            st.session_state.view_mode = "chat"  # Default view mode

    def _log_error(self, msg):
        """
        Centralized error logging function to log errors and display them in Streamlit.
        
        Args:
            msg (str): The error message to be logged and shown in the Streamlit app.
        
        Raises:
            Exception: If there is an error during the logging process.
        """
        self.logger.error(msg)
        st.error(msg)

    def delete_chat_history(self):
        """
        Deletes the chat history by making a request to the controller server's endpoint.
        
        Raises:
            requests.exceptions.RequestException: If there is an error in the POST request.
        """
        try:
            response = requests.post(f"{self.controller_url}/clean_conversations")
            response.raise_for_status()
            st.sidebar.success("Chat history deleted successfully.")
        except requests.exceptions.RequestException as e:
            self._log_error(f"Error deleting chat history: {e}")

    def is_numeric_claim(self, claim_text):
        """
        Checks if the claim consists only of numeric characters.
        
        Args:
            claim_text (str): The text of the claim to be checked.
        
        Returns:
            bool: True if the claim is numeric, False otherwise.
        """
        return claim_text.isdigit()

    def get_response(self, claim):
        """
        Requests a response from the controller server for the given claim.
        
        Args:
            claim (str): The claim text to be processed by the controller.
        
        Raises:
            requests.exceptions.RequestException: If there is an error in the POST request.
        
        Returns:
            dict: A dictionary containing the response, including title, summary, query result, sources, and images.
        """
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
        """
        Loads images from the specified folder and returns them as a list.
        
        Args:
            folder (str): The path to the folder containing image files.
        
        Returns:
            list: A list of PIL Image objects loaded from the folder.
        """
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

    def display_message(self, role, message, avatar=None):
        """
        Displays a chat message in the Streamlit interface.
        
        Args:
            role (str): The role of the sender, e.g., 'user' or 'assistant'.
            message (str): The content of the message to be displayed.
            avatar (str): The avatar to be used for the sender, default is 🦊.
        
        Raises:
            Exception: If there is an error in displaying the message.
        """
        if role == "assistant":
            chat_msg = st.chat_message("assistant", avatar="🦊")
        else:
            chat_msg = st.chat_message("user", avatar="👤")
        chat_msg.write(message)

    def display_claim_response(self, response):
        """
        Displays the response from the system regarding the user's claim, including title, summary, sources, and images.
        
        Args:
            response (dict): The response object containing claim title, summary, query result, sources, and images.
        
        Raises:
            Exception: If there is an error during the display of the claim response.
        """
        assistant_message = st.chat_message("assistant", avatar="🦊")

        # Styling for the title
        title_html = f"<h2 style='color: rgba(0, 0, 0, 0.9); font-size: 1.5em; line-height: 1.4;'>{response['title'][2:]}</h2>"
        assistant_message.markdown(title_html, unsafe_allow_html=True)

        # Styling for the summary
        summary_html = f"<p style='color: rgba(0, 0, 0, 0.6); font-size: 1.1em; line-height: 1.6;'>{response['summary']}</p>"
        assistant_message.markdown(summary_html, unsafe_allow_html=True)

        # Main response
        assistant_message.write(f"{response['response']}")

        # Display sources if available
        with assistant_message.expander("📌 Sources"):
            for src in response.get('sources', []):
                st.markdown(f"- [{src['title']}]({src['url']})")

        # Display images if available
        images = response.get('images', [])
        if images:
            with assistant_message.container():
                cols = st.columns(len(images))
                for col, img in zip(cols, images):
                    col.image(img)

    def get_conversations(self):
        """
        Retrieves all conversations from the controller server.
        
        Raises:
            requests.exceptions.RequestException: If there is an error in the GET request.
        
        Returns:
            list: A list of conversations.
        """
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
        """
        Displays a historical conversation in the chat interface.
        
        Args:
            conversation (dict): The conversation data to be displayed, including title, claim, answer, sources, and images.
        
        Raises:
            Exception: If there is an error in displaying the conversation.
        """
        def display_message_markdown(role, content, avatar=None):
            if role == "assistant":
                message = st.chat_message("assistant", avatar="🦊")
            else:
                message = st.chat_message("user", avatar="👤")
            message.markdown(content, unsafe_allow_html=True)

        # Concatenate title, claim, and response
        content = f"""
        <h2 style='color: rgba(0, 0, 0, 0.9); font-size: 1.5em; line-height: 1.4;'>{conversation['title']}</h2>
        """
        
        if conversation['claim']:
            content += f"<p style='color: rgba(0, 0, 0, 0.6); font-size: 1.1em; line-height: 1.6;'>{conversation['claim']}</p>"

        content += f"<p style='font-size: 1.1em; line-height: 1.6;'>{conversation['answer']}</p>"

        # Display the conversation
        display_message_markdown("assistant", content)

        # Display sources
        with st.expander("📌 Sources"):
            if conversation['sources']:
                sources_text = "\n\n".join([f"- [{src['title']}]({src['url']})" for src in conversation['sources']])
                st.markdown(sources_text)
            else:
                st.markdown("No sources available for this claim.")

        # Display images
        if conversation.get('images'):
            cols = st.columns(len(conversation['images']))
            for col, img in zip(cols, conversation['images']):
                col.image(img)

    def get_conversation_by_id(self, convo_id):
        """
        Retrieves a specific conversation by its ID.
        
        Args:
            convo_id (str): The ID of the conversation to be retrieved.
        
        Returns:
            dict: The conversation with the specified ID, or None if not found.
        """
        conversations = self.get_conversations()
        return next((convo for convo in conversations if convo['id'] == convo_id), None)

    def run(self):
        """
        Main function to run the Streamlit dashboard and manage the user interactions.
        
        Raises:
            Exception: If there is an error during the execution of the dashboard.
        """
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
                st.session_state.view_mode = "chat"
                st.session_state.messages = []
                st.session_state.selected_conversation = None
                st.rerun()

            if st.button("Exit Dashboard"):
                st.stop()

            st.image(self.image_sidebar)

            st.sidebar.title("Chat History")
            st.sidebar.text_input("Search claims...", key="search_query", placeholder="Type to search")

            # Get conversations and filter them
            conversations = self.get_conversations()
            filtered_conversations = [
                convo for convo in conversations if st.session_state.search_query.lower() in convo.get("title", "").lower()
            ] if "search_query" in st.session_state else conversations
            filtered_conversations = filtered_conversations[::-1]  # Reverse order to show the latest first

            # Initialize state for toggling view
            if 'show_all_conversations' not in st.session_state:
                st.session_state.show_all_conversations = False

            # Limit the number of conversations to display
            max_display = 5
            if not st.session_state.show_all_conversations:
                display_conversations = filtered_conversations[:max_display]
            else:
                display_conversations = filtered_conversations

            # Display the conversations
            for i, convo in enumerate(display_conversations):
                if st.sidebar.button(convo['title'][:50] + ("..." if len(convo['title']) > 50 else ""), key=f"convo_{i}"):
                    st.session_state.view_mode = "history"
                    st.session_state.selected_conversation = convo
                    st.rerun()

            # Show toggle button for conversations
            if len(filtered_conversations) > max_display:
                if st.sidebar.button(
                    "Show Less Conversations" if st.session_state.show_all_conversations else "Show More Conversations"
                ):
                    st.session_state.show_all_conversations = not st.session_state.show_all_conversations
                    st.rerun()

if __name__ == "__main__":
    dashboard = DashboardPipeline()
    dashboard.run()