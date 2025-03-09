import pandas as pd
from sklearn.datasets import make_classification

# Generate synthetic data for classification
X, y = make_classification(
    n_samples=200,         # Number of samples (data points)
    n_features=2,          # Number of features (input variables)
    n_classes=2,           # Number of classes (binary classification: 0 or 1)
    n_informative=2,       # Number of informative features (in this case, all features are informative)
    n_redundant=0,         # Number of redundant features (none in this case)
    random_state=42        # Random seed for reproducibility
)

# Create a DataFrame with the features (X) and labels (y)
df = pd.DataFrame(X, columns=["feature1", "feature2"])
df["target"] = y  # Add the target variable (classification label) as a column

# Save the DataFrame to a CSV file named "data.csv"
df.to_csv("data.csv", index=False)
print("Data has been generated and saved to data.csv.")
