import os
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from LiveExecution.src.config import load_config
from LiveExecution.src.models import ModelLoader

def test():
    load_dotenv()
    try:
        config = load_config()
        print("Config loaded successfully.")
        loader = ModelLoader()
        if loader.load_all_models():
            print("Models loaded successfully.")
        else:
            print("Failed to load models.")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test()
