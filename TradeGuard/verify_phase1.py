from src.generate_dataset import DatasetGenerator
from pathlib import Path
import sys

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

def verify():
    print("Verifying Phase 1...")
    gen = DatasetGenerator()
    
    # Test Data Loading (Point to backtest data)
    gen.data_dir = Path("Alpha/backtest/data")
    print(f"Loading data from {gen.data_dir}...")
    data = gen.load_data()
    print(f"Loaded {len(data)} assets.")
    for asset, df in data.items():
        print(f"  {asset}: {len(df)} rows")
        
    # Test Model Loading
    model_path = "Alpha/models/checkpoints/8.03.zip"
    if not Path(model_path).exists():
        model_path = "checkpoints/8.03.zip"
        
    print(f"Loading model from {model_path}...")
    model = gen.load_model(model_path)
    if model:
        print("Model loaded successfully.")
    else:
        print("Model load failed.")
        
    print("Verification Script Complete.")

if __name__ == "__main__":
    verify()
