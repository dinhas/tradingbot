import os
import sys
import time
import subprocess
from pyngrok import ngrok
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run_tensorboard_ngrok(log_dir="./logs", port=6006):
    """
    Starts TensorBoard and exposes it via ngrok.
    """
    # 1. Check if log_dir exists
    if not os.path.exists(log_dir):
        print(f"Warning: Log directory '{log_dir}' does not exist yet. TensorBoard might be empty.")
        os.makedirs(log_dir, exist_ok=True)

    # 2. Authenticate ngrok (if not already done)
    # You can set the token via environment variable NGROK_AUTHTOKEN
    # or it will ask for input if not found in config.
    
    # Check if token is configured
    try:
        # This might fail if no token is set in the config file
        # We can check if we need to set it.
        pass 
    except Exception:
        pass

    print("--- TensorBoard & Ngrok Launcher ---")
    
    # Optional: Allow user to input token if not set
    # For automation, best to set NGROK_AUTHTOKEN env var before running.
    if "NGROK_AUTHTOKEN" in os.environ:
        ngrok.set_auth_token(os.environ["NGROK_AUTHTOKEN"])
        print("Using NGROK_AUTHTOKEN from environment.")
    
    # 3. Start TensorBoard
    print(f"Starting TensorBoard on port {port} reading from {log_dir}...")
    # We run it in the background
    tb_process = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give it a moment to start
    time.sleep(3)
    
    if tb_process.poll() is not None:
        print("Error: TensorBoard failed to start.")
        print(tb_process.stderr.read().decode())
        return

    print("TensorBoard started.")

    # 4. Start ngrok tunnel
    print("Opening ngrok tunnel...")
    try:
        public_url = ngrok.connect(port).public_url
        print(f"\n✅ TensorBoard is live at: {public_url}\n")
        print("Press Ctrl+C to stop.")
    except Exception as e:
        print(f"Error connecting ngrok: {e}")
        print("Make sure you have set your ngrok authtoken using: ngrok config add-authtoken <TOKEN>")
        print("Or set the NGROK_AUTHTOKEN environment variable.")
        tb_process.terminate()
        return

    # 5. Keep alive
    try:
        while True:
            time.sleep(1)
            if tb_process.poll() is not None:
                print("TensorBoard process ended unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Closing ngrok tunnel...")
        ngrok.disconnect(public_url)
        print("Terminating TensorBoard...")
        tb_process.terminate()

if __name__ == "__main__":
    # Default to ../logs if running from util folder, or ./logs if running from root
    # Assuming script is in util/ and run from root or util
    
    # If run from root (e.g. python util/run_tensorboard.py)
    if os.path.exists("logs"):
        LOG_DIR = "logs"
    # If run from util (e.g. cd util; python run_tensorboard.py)
    elif os.path.exists("../logs"):
        LOG_DIR = "../logs"
    else:
        # Default fallback
        LOG_DIR = "logs"
        
    # Allow overriding via command line
    if len(sys.argv) > 1:
        LOG_DIR = sys.argv[1]

    run_tensorboard_ngrok(LOG_DIR)
