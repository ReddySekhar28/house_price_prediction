import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "house_price_train.csv")

df = pd.read_csv(data_path)

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# -------------------------------
# Data Preprocessing
# -------------------------------

# Drop unnecessary columns if present
drop_cols = ['id', 'date']  # adjust if needed
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# Create derived features (same as app.py)
if 'date' in df.columns:
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
else:
    df['year'] = 2014
    df['month'] = 5

# Features & Target
X = df.drop("price", axis=1)
y = df["price"]

# Convert categorical features
X = pd.get_dummies(X)

# Save column names
columns = X.columns

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Model Training
# -------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")

# -------------------------------
# Save Model & Files
# -------------------------------
pickle.dump(model, open(os.path.join(BASE_DIR, "model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(BASE_DIR, "scaler.pkl"), "wb"))
pickle.dump(columns, open(os.path.join(BASE_DIR, "columns.pkl"), "wb"))

print("Model, Scaler, and Columns saved successfully!")