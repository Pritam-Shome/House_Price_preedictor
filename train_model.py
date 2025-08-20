import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("house_price_dataset.csv")

# Encode categorical columns if needed
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Features (X) and target (y)
X = data.drop("Price", axis=1)   # assuming target column is "Price"
y = data["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and feature names
with open("model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)

print("✅ Model saved successfully as model.pkl")
