# Product Guidelines

## Documentation & Coding Style
This project adopts a **Hybrid Technical Style** that balances academic rigor with pragmatic engineering and transparent research documentation.

### 1. Code: Pragmatic & Technical
*   **Focus:** Implementation details should be concise and practical.
*   **Comments:** Focus on the *why* and *how* of complex logic, not just the *what*.
*   **Typing:** Strong typing (Python `typing` module) is mandatory for all function signatures to ensure clarity and safety.
*   **Structure:** Follow standard Python/ML project structures (separating data, src, models, logs).

### 2. Models & Math: Strict & Academic
*   **Definitions:** Mathematical concepts (e.g., PPO loss functions, LightGBM gradients, Sharpe Ratio) must be defined with statistical precision.
*   **Metrics:** Success metrics (Precision, Drawdown, etc.) are treated as strict statistical targets, not vague goals.
*   **References:** Cite relevant papers or mathematical derivations when implementing complex algorithms.

### 3. Research & Experiments: Exploratory
*   **Lab Notebook Approach:** Document the evolution of the models. Record hypotheses, experiment setups, results, and failures.
*   **Narrative:** It is encouraged to write detailed narratives explaining the "trial-and-error" process of model development, helping to contextualize why certain decisions were made.
*   **Transparency:** Be honest about what didn't work. Negative results in backtesting are as valuable as positive ones.

## Visual Identity
*   **Data-First:** The primary "visuals" of this project are charts, logs, and performance graphs (TensorBoard, Matplotlib).
*   **Clarity:** Visualizations should be clean, labeled clearly, and focused on conveying statistical truth without unnecessary decoration.
