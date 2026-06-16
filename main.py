from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle

class HouseFeatures(BaseModel):
    overall_qual: int
    overall_cond: int
    effective_year: int
    x1st_flr_sf: int
    x2nd_flr_sf: int
    garage_cars: int
    full_bath: int
    neighborhood_tier: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts once at startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

df = pd.read_csv("ames_model.csv")

@app.get("/")
def root():
    return {"status": "House Price API is running"}

@app.get("/api/cleaning-summary")
def cleaning_summary():
    return [
        {"step": "Raw dataset", "rows": 2930, "removed": 0, "reason": "Original ames_raw from AmesHousing package"},
        {"step": "Drop identifiers", "rows": 2930, "removed": 0, "reason": "Removed Order and PID — no predictive value"},
        {"step": "Drop rows with genuine NAs", "rows": 2925, "removed": 5, "reason": "Electrical, basement and garage measurements with true missing values"},
        {"step": "Impute lot_frontage", "rows": 2925, "removed": 0, "reason": "Median imputation by neighborhood and lot configuration"},
        {"step": "Remove abnormal sales", "rows": 2770, "removed": 155, "reason": "Non-arms-length transactions per dataset author's recommendation"},
        {"step": "Remove outliers", "rows": 2410, "removed": 360, "reason": "Above ground living area > 4,000 sq ft distort the model"},
    ]

@app.get("/api/model-comparison")
def model_comparison():
    return [
        {"model": "Simple Regression", "variables": 1, "r2": 0.66, "rmse": 0.2149, "notes": "overall_qual only"},
        {"model": "Multiple Regression", "variables": 103, "r2": 0.94, "rmse": 0.0973, "notes": "All features"},
        {"model": "Ridge", "variables": 103, "r2": None, "rmse": 0.0993, "notes": "All features shrunk"},
        {"model": "Lasso", "variables": 92, "r2": 0.94, "rmse": 0.0971, "notes": "12 variables zeroed out — champion model"},
    ]

@app.get("/api/correlation")
def correlation():
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()["log_sale_price"].drop("log_sale_price")
    corr = corr.drop("sale_price", errors="ignore")
    corr_sorted = corr.abs().sort_values(ascending=False).head(15)
    return [
        {"variable": k, "correlation": round(float(corr[k]), 3)}
        for k in corr_sorted.index
    ]

@app.get("/api/sale-price-distribution")
def sale_price_distribution():
    return df["sale_price"].tolist()

@app.get("/api/neighborhood-tiers")
def neighborhood_tiers():
    # Group by neighborhood tier — tier assigned via k-means in R
    grouped = (
        df.groupby("neighborhood_tier")["sale_price"]
        .agg(median_price="median", count="count")
        .reset_index()
    )
    grouped["median_price"] = grouped["median_price"].round(0).astype(int)
    # Define order for display
    tier_order = ["Budget", "Mid-Range", "Premium", "Luxury"]
    grouped["neighborhood_tier"] = pd.Categorical(
        grouped["neighborhood_tier"], categories=tier_order, ordered=True
    )
    grouped = grouped.sort_values("neighborhood_tier")
    return grouped.rename(columns={"neighborhood_tier": "tier"}).to_dict(orient="records")

@app.get("/api/quality-vs-price")
def quality_vs_price():
    grouped = (
        df.groupby("overall_qual")["sale_price"]
        .agg(median_price="median", count="count")
        .reset_index()
    )
    grouped["median_price"] = grouped["median_price"].round(0).astype(int)
    grouped.columns = ["quality", "median_price", "count"]
    return grouped.to_dict(orient="records")

@app.get("/api/living-area-vs-price")
def living_area_vs_price():
    sample = df.copy()
    sample["living_area"] = sample["x1st_flr_sf"] + sample["x2nd_flr_sf"]
    return sample[["living_area", "sale_price", "neighborhood_tier"]].to_dict(orient="records")

@app.get("/api/year-built-vs-price")
def year_built_vs_price():
    return df[["effective_year", "sale_price"]].to_dict(orient="records")

@app.post("/api/predict")
def predict(features: HouseFeatures):
    # Start with median values for all numeric features
    # This means unspecified features default to "typical house"
    numeric_cols = df.drop(columns=["sale_price", "log_sale_price", "neighborhood_tier"]).select_dtypes(include=[np.number])
    
    # Get dummies to match training features
    df_dummies = pd.get_dummies(df.drop(columns=["sale_price", "log_sale_price"]), drop_first=True)
    
    # Start from median row
    input_dict = df_dummies.median().to_dict()
    
    # Zero out dummy columns then set the right one
    tier_cols = [c for c in input_dict if c.startswith("neighborhood_tier_")]
    for col in tier_cols:
        input_dict[col] = 0
    tier_col = f"neighborhood_tier_{features.neighborhood_tier}"
    if tier_col in input_dict:
        input_dict[tier_col] = 1

    # Override with user inputs
    input_dict["overall_qual"]  = features.overall_qual
    input_dict["overall_cond"]  = features.overall_cond
    input_dict["effective_year"] = features.effective_year
    input_dict["x1st_flr_sf"]   = features.x1st_flr_sf
    input_dict["x2nd_flr_sf"]   = features.x2nd_flr_sf
    input_dict["garage_cars"]    = features.garage_cars
    input_dict["full_bath"]      = features.full_bath

    # Align columns to training feature order
    input_df = pd.DataFrame([input_dict])[feature_names]
    input_scaled = scaler.transform(input_df)
    log_price = model.predict(input_scaled)[0]
    predicted_price = int(np.exp(log_price))

    return {
        "predicted_price": predicted_price,
        "log_price": round(float(log_price), 4)
    }

@app.get("/api/sale-price-stats")
def sale_price_stats():
    return {
        "mean": round(float(df["sale_price"].mean()), 2),
        "std": round(float(df["sale_price"].std()), 2),
        "log_mean": round(float(df["log_sale_price"].mean()), 4),
        "log_std": round(float(df["log_sale_price"].std()), 4),
        "log_prices": df["log_sale_price"].tolist()
    }