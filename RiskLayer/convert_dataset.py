import pandas as pd
import numpy as np
import os

SL_CHOICES = [0.5, 0.75, 1.0, 1.5, 1.75, 2.0, 2.5, 2.75, 3.0]


def convert_sl_to_rl_dataset(input_path, output_path):
    print(f"Loading dataset from {input_path}...")
    df = pd.read_parquet(input_path)

    print(f"Converting {len(df)} samples...")

    def sl_mult_to_idx(sl_mult):
        idx = np.argmin(np.abs(np.array(SL_CHOICES) - sl_mult))
        return idx

    df["target_sl_idx"] = df["target_sl_mult"].apply(sl_mult_to_idx)

    df = df.rename(
        columns={
            "target_sl_mult": "target_sl_mult_old",
            "target_tp_mult": "target_tp_mult",
            "target_size": "target_size",
        }
    )

    df["target_sl_idx"] = df["target_sl_idx"].astype(np.int64)
    df["target_tp_mult"] = df["target_tp_mult"].astype(np.float32)
    df["target_size"] = df["target_size"].astype(np.float32)

    output_df = df[
        [
            "timestamp",
            "asset",
            "direction",
            "entry_price",
            "atr",
            "features",
            "target_sl_idx",
            "target_tp_mult",
            "target_size",
        ]
    ].copy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_parquet(output_path, index=False)

    print(f"RL dataset saved to {output_path}")
    print(f"Total samples: {len(output_df)}")
    print(f"Columns: {list(output_df.columns)}")

    print("\nTarget distribution (SL idx):")
    print(output_df["target_sl_idx"].value_counts().sort_index())


if __name__ == "__main__":
    input_path = "RiskLayer/data/sl_risk_dataset.parquet"
    output_path = "RiskLayer/data/rl_risk_dataset.parquet"
    convert_sl_to_rl_dataset(input_path, output_path)
