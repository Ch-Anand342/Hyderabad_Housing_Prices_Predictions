import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# LOAD DATASET

df = pd.read_csv("Hyderbad_House_price.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

# DATA CLEANING

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Extract BHK from title

df["bhk"] = df["title"].str.extract(r'(\d+)')
df["bhk"] = pd.to_numeric(df["bhk"], errors="coerce")
df["bhk"] = df["bhk"].fillna(df["bhk"].median())
df["bhk"] = df["bhk"].astype(int)

print("\nMissing Values")
print(df.isnull().sum())

print("\nColumns:")
print(df.columns)

# SAVE TO SQL DATABASE

conn = sqlite3.connect("hyderabad.db")

df.to_sql(
    "houses",
    conn,
    if_exists="replace",
    index=False
)

print("\nData Stored in SQL Database")

# NUMPY ANALYSIS

print(
    "Average Price:",
    round(np.mean(df["price(L)"]), 2),
    "Lakhs"
)

print(
    "Maximum Price:",
    np.max(df["price(L)"]),
    "Lakhs"
)

print(
    "Minimum Price:",
    np.min(df["price(L)"]),
    "Lakhs"
)

print(
    "Median Price:",
    np.median(df["price(L)"]),
    "Lakhs"
)

# TOP LOCALITIES

locality_prices = (
    df.groupby("location")["price(L)"]
      .mean()
      .sort_values(ascending=False)
      .head(10)
)

print("\nTop 10 Expensive Localities")

print(locality_prices)

### EXPORT TO EXCEL

# Complete cleaned dataset
df.to_excel(
    "House_Report.xlsx",
    index=False
)

# Summary report
summary = (
    df.groupby("location")
      .agg({
          "price(L)": "mean",
          "area_insqft": "mean",
          "bhk": "mean"
      })
      .round(2)
)

summary.to_excel(
    "House_Report_Summary.xlsx"
)

# VISUALIZATION

plt.figure(figsize=(12, 6))

locality_prices.plot(kind="bar")

plt.title(
    "Top 10 Localities by Average House Price"
)

plt.xlabel("Location")
plt.ylabel("Average Price (Lakhs)")

plt.tight_layout()

plt.savefig("locality_prices.png")

plt.show()

print("\nChart Saved")

# MACHINE LEARNING

print("\nTraining Model...")

# Encode categorical columns

location_encoder = LabelEncoder()
status_encoder = LabelEncoder()

df["location_encoded"] = (
    location_encoder.fit_transform(
        df["location"]
    )
)

df["status_encoded"] = (
    status_encoder.fit_transform(
        df["building_status"]
    )
)

# Features

X = df[
    [
        "area_insqft",
        "bhk",
        "rate_persqft",
        "location_encoded",
        "status_encoded"
    ]
]

# Target

y = df["price(L)"]

# Split

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

# Model

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# Prediction

predictions = model.predict(
    X_test
)

# Accuracy

r2 = r2_score(
    y_test,
    predictions
)

mae = mean_absolute_error(
    y_test,
    predictions
)

print("\n===== MODEL PERFORMANCE =====")

print(
    "R2 Score:",
    round(r2, 3)
)

print(
    "Mean Absolute Error:",
    round(mae, 2)
)

# Save model

joblib.dump(
    model,
    "house_price_model.pkl"
)

print("\nModel Saved")

# SAMPLE PREDICTION

print("\n===== HOUSE PRICE PREDICTION =====")

print("\nAvailable Locations:")

for loc in df["location"].unique()[:20]:
    print(loc)

location_name = input(
    "\nEnter Location: "
)

try:

    location_encoded = (
        location_encoder.transform(
            [location_name]
        )[0]
    )

    status_encoded = (
        status_encoder.transform(
            ["Under Construction"]
        )[0]
    )

    area = float(
        input(
            "Enter Area in Sqft: "
        )
    )

    bhk = int(
        input(
            "Enter BHK: "
        )
    )

    rate = float(
        input(
            "Enter Rate Per Sqft: "
        )
    )

    predicted_price = (
        model.predict(
            [[
                area,
                bhk,
                rate,
                location_encoded,
                status_encoded
            ]]
        )
    )

    print(
        f"\nPredicted House Price = {predicted_price[0]:.2f} Lakhs"
    )

except:

    print(
        "\nLocation not found in dataset."
    )

conn.close()

print("\nPROJECT COMPLETED SUCCESSFULLY")
