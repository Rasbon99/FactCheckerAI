![Logo](assets/Logo.png)

**Fact Online eXamination AI** (FOX AI) is an advanced application designed to evaluate the reliability of a news item through state-of-the-art deep fact-checking techniques, leveraging highly credible sources.

Beginning with a claim provided by the user, related news articles are retrieved and assessed based on the reliability of their sources. 

To ensure high accuracy in source selection, the system performs a dual filtering process:

1. **NewsGuard Ranking Database**: Sources are initially filtered using the NewsGuard Ranking Database to prioritize reliability.
2. **Correlation Testing**: An additional correlation test is conducted using an LLM to verify the relevance and alignment between the source and the claim to be validated. This step minimizes the risk of selecting irrelevant or misleading articles.

These articles are then processed and linked within a GraphRAG framework. The Large Language Model (LLM) generates a contextual response to the claim.

**Objectives**:

- **Truthfulness Assessment**: Determine the truthfulness of the analyzed news item based on the identified sources.
- **Transparent Explanations**: Provide clear and detailed explanations of the classification process, explicitly citing the sources used in the evaluation.
- **Knowledge Graphs**: Generate knowledge graphs from the identified sources to enhance interpretability.
- **Comprehensive Reporting**: Deliver a user-friendly, interactive report accessible via a dashboard.

### Tools and Technologies

- **Dashboard**: Built using **Streamlit** to provide an interactive and intuitive user interface.
- **Large Language Models (LLMs)**: Hosted on **Groq Cloud** and **Ollama** for advanced reasoning and validation tasks.
- **GraphRAG Framework**: Utilizes **Neo4j** for constructing and analyzing relational knowledge graphs.

---

## Prerequisites
To use this project, you need to configure the following tools:

1. **Docker Setup** *(Recommended)*
   - If using Docker, you can skip the manual installation steps below.
2. **Python Libraries** *(Manual Installation Only)*
3. **Neo4j** *(Manual Installation Only)*
4. **Ollama** *(Manual Installation Only)*
5. **NewsGuard Ranking Database** *(Recommended)*
6. **Groq Cloud**

---

## Option 1: Run with Docker *(Recommended)*

To simplify the setup process, you can run the application using Docker. There are two options depending on your system:

### Standard Docker Setup
For systems that do not have a GPU compatible with CUDA, use the following command to start the application:
```bash
docker compose up --build
```

### GPU-Accelerated Setup (NVIDIA GPUs Required)
If your system has an NVIDIA GPU and supports CUDA, you can use the GPU-accelerated version. Ensure that you have the **NVIDIA Container Toolkit** installed before running the following command:

- **Installation Guide:** [NVIDIA Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

- If you are using **Docker Desktop on Windows**, the NVIDIA Container Toolkit is already included.

Once the toolkit is installed, run:
```bash
docker-compose -f docker-compose-gpu.yml build
docker-compose -f docker-compose-gpu.yml up
```

Note: Neo4j authentication is disabled in the Docker version, so you do not need to provide a username and password.

---

## Option 2: Manual Installation
If you prefer to install and run the application locally without Docker, follow these steps:

### Step 1: Set Up the Environment

#### 1. Create and activate a new virtual environment:
```bash
conda create --name myenv python=3.13.1
conda activate myenv
```

#### 2. Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Step 2: Neo4j Setup

#### Download Neo4j
Download the Community Edition of Neo4j Graph Database Self-Managed from the following link: [Neo4j Deployment Center](https://neo4j.com/deployment-center/).

Note: In the local version, you need to set the username and password for Neo4j. The default admin credentials are neo4j for both the username and password.

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
```

#### Add Environment Variables for Llama
- **Windows:**
  1. Follow the instructions in the "Add Environment Variables" section and the installation path of Ollama.

### Step 4: NewsGuard Ranking Database *(Recommended)*
If available, request access to the NewsGuard Ranking Database API by contacting their team. Once you receive the credentials.

### Step 5: Register on Groq Cloud
1. Register on [Groq Cloud](https://console.groq.com/).
2. After registration, generate an API key and store it securely.

### Step 6: Create the `key.env` File

In case of launching with Docker, set `DOCKER=true` and uncomment all variables under the **Docker Version** section. Otherwise, set `DOCKER=false` and uncomment the variables under the **Local Version** section.

```env
DOCKER=true

# API URL Docker Version
OLLAMA_SERVER_URL=http://ollama:11434
NEO4J_SERVER_URL=http://neo4j:7474
OLLAMA_API_URL=http://ollama:11434
NEO4J_API_URL=http://neo4j:7474
BACKEND_API_URL=http://backend:8001
CONTROLLER_API_URL=http://controller:8003
NEO4J_URI=bolt://neo4j:7687

# API URL Local Version
# OLLAMA_SERVER_URL=http://localhost:11434
# NEO4J_SERVER_URL=http://localhost:7474
# OLLAMA_API_URL=http://localhost:8000
# NEO4J_API_URL=http://localhost:8002
# BACKEND_API_URL=http://localhost:8001
# CONTROLLER_API_URL=http://localhost:8003
# NEO4J_URI=bolt://localhost:7687

# DASHBOARD CONSTANTS
LOG_FILE=app.log
AI_IMAGE_UI=Dashboard/FOX_AI.png

# NEWSGUARD VARIABLES
CLIENT_API_ID =
NG_API_KEY = 

# DATABASE VARIABLES
SQLDB_PATH=data/fact_checker.db
ASSET_PATH=assets

# GRAPHRAG VARIABLES
MODEL_LLM_NEO4J = phi3.5:latest
NEO4J_USERNAME = ''
NEO4J_PASSWORD = ''

# GROQ VARIABLES
GROQ_MODEL_NAME=llama-3.3-70b-versatile
GROQ_LOW_MODEL_NAME=gemma2-9b-it
GROQ_API_KEY=
```

## Authors

- [Leonardo Catello](https://github.com/Leonard2310)
- [Lorenzo Manco](https://github.com/Rasbon99)
- [Carmine Grosso](https://github.com/httpix3l)
- [Aurora D'Ambrosio](https://github.com/AuroraD-99)
- [Gennaro Iannicelli](https://github.com/Gennaro2806)

---

## Credits
We would like to express our sincere gratitude to the NewsGuard team for granting access to the NewsGuard News Reliability Rating Database. This database has been a crucial resource in enhancing the quality and reliability of the data in our Fact-Checking project. 

For more information about NewsGuard and their work, visit their official website: https://www.newsguardtech.com.

---

## License
This project is licensed under the [GNU General Public License v3.0](LICENSE). Refer to the LICENSE file for more information.