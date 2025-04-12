import numpy as np
import pandas as pd

def generate_dataset(n_samples=1000, seed=42):
    np.random.seed(seed)
    a = np.random.uniform(1, 100, n_samples)
    b = np.random.uniform(1, 100, n_samples)
    c = np.sqrt(a**2 + b**2)
    
    df = pd.DataFrame({'a': a, 'b': b, 'c': c})
    return df

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/triangle_dataset.csv", index=False)
    print("Dataset saved to data/triangle_dataset.csv")


