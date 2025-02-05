# Fact Online eXamination AI

## Prerequisites
To use this project, you need to configure the following tools:

1. **Python Libraries** 
2. **Neo4j**
3. **Ollama**
4. **NewsGuard Ranking Database**
5. **Groq Cloud**

### Step 1: Set Up the Environment
1. Create a new Python virtual environment:
   ```bash
   python -m venv env
   ```
2. Activate the virtual environment:
   - **Windows:**
     ```bash
     .\env\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     source env/bin/activate
     ```
3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Neo4j Setup

#### Download Neo4j
Download Neo4j from the following link: [Neo4j Deployment Center](https://neo4j.com/deployment-center/).

#### Configure APOC
1. Copy the `apoc-5.26.1-core.jar` file from the `labs` folder and paste it into the `plugins` folder. Rename the copied file to `apoc.jar`.
2. Edit the `neo4j.conf` file in the `conf` folder and add the following lines at the end of the file:

   ```
   # Configure the plugin directory
   server.directories.plugins=plugins

   # Enable APOC procedures
   dbms.security.procedures.unrestricted=apoc.*, algo.*

   dbms.security.procedures.allowlist=apoc.meta.data,apoc.help
   ```

#### Set Up Environment Variables
1. Add the path to the `bin` folder of Neo4j to your system's environment variables:

   - **Windows:**
     1. Open the Start menu and search for "Environment Variables".
     2. In the "System Properties" window, click on "Environment Variables".
     3. Under "System Variables", click "New" and add:
        - Value: `C:\Path\To\Bin`
   
   - **Mac:**
     1. Open the terminal.
     2. Edit the `~/.zshrc` or `~/.bash_profile` file (depending on your shell) by adding:
        ```
        export NEO4J_BIN=/path/to/bin/folder
        ```
     3. Save the file and reload the configuration by running:
        ```bash
        source ~/.zshrc
        ```

### Step 3: Ollama Setup

#### Download Ollama
1. Visit the official website: [Ollama](https://ollama.com/) and download the software for your platform.

#### Installation on Windows
Windows users need to use the Linux version of Ollama within WSL (Windows Subsystem for Linux). Follow the official instructions to set up WSL if it's not already installed.

#### Download Models from Ollama
After installation, download the desired models from the official registry using the following commands, we recommend:

```bash
ollama pull phi3.5
ollama pull gemma2-9b-it
```

#### Add Environment Variables for Llama
- **Windows:**
  1. Follow the instructions in the "Add Environment Variables" section and the installation path of Ollama.

### Step 4: NewsGuard Ranking Database
If available, request access to the NewsGuard Ranking Database by contacting their team. Once you receive the credentials, add the `CLIENT_API_ID` and `NG_API_KEY` variables to your environment configuration.

### Step 5: Register on GROQ Cloud
1. Register on [GROQ Cloud](https://www.groq.com/).
2. After registration, generate an API key and store it securely.

### Step 6: Create the `key.env` File
Add all the required environment variables to a file named `key.env` in the following format:

```env
# NEWSGUARD VARIABLES
CLIENT_API_ID=
NG_API_KEY=

# DATABASE VARIABLES
SQLDB_PATH=Database/data/fact_checker.db

# GRAPHRAG VARIABLES
MODEL_LLM_NEO4J=phi3.5:latest
MODEL_LLM_NER=gemma:latest
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=
NEO4J_PASSWORD=

# GROQ VARIABLES
GROQ_MODEL_NAME=llama-3.3-70b-versatile
GROQ_LOW_MODEL_NAME=gemma2-9b-it
GROQ_API_KEY=

# DASHBOARD CONSTANTS
LOG_FILE=app.log
AI_IMAGE_UI=Dashboard/AI_IMAGE.png
CLAIM_PROCESSING_TIME=5
```

## Credits
We would like to express our sincere gratitude to the NewsGuard team for granting access to the NewsGuard News Reliability Rating Database. This database has been a crucial resource in enhancing the quality and reliability of the data in our Fact-Checking project. 

For more information about NewsGuard and their work, visit their official website: https://www.newsguardtech.com.