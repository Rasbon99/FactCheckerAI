import os
import subprocess
import dotenv


def get_model_config():
    """Determines which model to use based on the environment."""
    env = os.getenv("DEPLOYMENT_ENV", "local").lower()

    if env == "cluster":
        # The heavy 70B model for Cluster (Requires ~40GB VRAM)
        return {
            "repo_id": "bartowski/Meta-Llama-3-70B-Instruct-GGUF",
            "filename": "Meta-Llama-3-70B-Instruct-Q4_K_M.gguf",
        }
    else:
        # The lightweight 8B model for local testing (Requires ~5GB RAM)
        return {
            "repo_id": "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
            "filename": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        }


def ensure_model_downloaded(repo_id, filename):
    """Checks if the model exists locally; downloads it if missing."""
    model_dir = "./models"
    model_path = os.path.join(model_dir, filename)

    if os.path.exists(model_path):
        print(f"✅ Model found locally: {filename}")
        return model_path

    print(f"⏳ Model not found. Downloading {filename} from Hugging Face...")
    print("This may take a few minutes depending on your internet connection.")

    # Calls the same 'hf download' command you ran in your terminal
    command = ["hf", "download", repo_id, filename, "--local-dir", model_dir]

    try:
        subprocess.run(command, check=True)
        print("✅ Download complete!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading the model: {e}")
        exit(1)

    return model_path


def start_server():
    """Manages the download process and boots the llama.cpp server."""
    dotenv.load_dotenv("key.env", override=True)

    # 1. Figure out which model we need
    config = get_model_config()
    port = os.getenv("LLM_MODEL_PORT", "8080")

    print(f"🌍 Environment detected: {os.getenv('DEPLOYMENT_ENV', 'local').upper()}")

    # 2. Ensure it is downloaded
    model_path = ensure_model_downloaded(config["repo_id"], config["filename"])

    # 3. Build the server command
    command = [
        "llama-server",
        "-m",
        model_path,
        "--port",
        port,
        "--n-gpu-layers",
        "99",  # Automatically utilizes Mac Metal GPU or Cluster NVIDIA GPUs
    ]

    print(f"🚀 Starting local llama.cpp server on http://localhost:{port}...")
    try:
        # Boot the server in the foreground
        subprocess.run(command)
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped manually.")
    except FileNotFoundError:
        print(
            "❌ Error: 'llama-server' command not found. Ensure llama.cpp is installed."
        )


if __name__ == "__main__":
    start_server()
