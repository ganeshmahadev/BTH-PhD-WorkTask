import numpy as np
import pandas as pd
if __name__ == "__main__":
# Generate synthetic dataset
    np.random.seed(42)
    num_samples = 500
    df = pd.DataFrame(
    {
    "Category1": np.random.choice(["A", "B", "C", "D", "E"],
    num_samples, p=[0.2, 0.4, 0.2, 0.1, 0.1]),
    "Value1": np.random.normal(10, 2,
        num_samples), # Continuous variable
    "Value2": np.random.normal(20, 6,
        num_samples), # Continuous variable
    }
)
df.to_csv("dataset.csv", sep=";")