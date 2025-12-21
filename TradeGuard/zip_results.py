import shutil
import os
from pathlib import Path

def zip_results():
    # Detect environment
    is_kaggle = os.path.exists('/kaggle/working')
    
    if is_kaggle:
        base_path = Path("/kaggle/working/TradingBot/TradeGuard/models")
        if not base_path.exists():
            # Try alternative path if TradingBot is lowercase or missing
            base_path = Path("/kaggle/working/tradingbot/TradeGuard/models")
            
        output_filename = "/kaggle/working/tradeguard_results"
    else:
        # Local path
        base_path = Path("TradeGuard/models")
        output_filename = "tradeguard_results"

    if not base_path.exists():
        print(f"Error: Directory {base_path} not found.")
        return

    print(f"Zipping contents of {base_path}...")
    shutil.make_archive(output_filename, 'zip', base_path)
    print(f"Created {output_filename}.zip successfully!")

if __name__ == "__main__":
    zip_results()
