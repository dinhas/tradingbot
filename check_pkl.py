
import pickle
from stable_baselines3.common.vec_env import VecNormalize

path = r"e:\tradingbot\models\checkpoints\risk\model10M.pkl"
try:
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Type of loaded data: {type(data)}")
except Exception as e:
    print(f"Error: {e}")
