
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import logging

app = FastAPI()

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "backtest" / "data"
RESULTS_DIR = BASE_DIR.parent.parent / "backtest" / "results"

# Setup Static and Templates
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

logging.basicConfig(level=logging.INFO)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/datasets")
async def get_datasets():
    # List Parquet files (Assets)
    assets = []
    if DATA_DIR.exists():
        for f in DATA_DIR.iterdir():
            if f.name.endswith(".parquet") and "5m_2025" in f.name:
                asset_name = f.name.replace("_5m_2025.parquet", "")
                assets.append(asset_name)
    
    # List CSV files (Backtest Results)
    results = []
    if RESULTS_DIR.exists():
        for f in RESULTS_DIR.iterdir():
            if f.name.endswith(".csv"):
                results.append(f.name)
                
    return {"assets": sorted(assets), "results": sorted(results, reverse=True)}

@app.get("/api/candles")
async def get_candles(asset: str):
    # Load Parquet
    fname = DATA_DIR / f"{asset}_5m_2025.parquet"
    if not fname.exists():
        raise HTTPException(status_code=404, detail="Asset data not found")
    
    try:
        df = pd.read_parquet(fname)
        # Convert to list of dicts: {time: seconds, open, high, low, close, volume}
        # Lightweight charts expects seconds for time
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'time'}, inplace=True)
        # Ensure time is seconds timestamp
        df['time'] = df['time'].astype('int64') // 10**9
        
        data = df[['time', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')
        return data
    except Exception as e:
        logging.error(f"Error loading candles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trades")
async def get_trades(result_file: str, asset: str):
    fname = RESULTS_DIR / result_file
    if not fname.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
        
    try:
        df = pd.read_csv(fname)
        
        # Filter by asset
        df = df[df['asset'] == asset].copy()
        
        if df.empty:
            return []
            
        # Parse Dates
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['time'] = df['timestamp'].astype('int64') // 10**9
        
        # Convert to dict
        trades = df.to_dict(orient='records')
        return trades
    except Exception as e:
        logging.error(f"Error loading trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
