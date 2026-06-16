# House Price Prediction API

FastAPI backend serving house price prediction results for the House Price Prediction case study.

## Overview

REST API that trains a Lasso regression model on the Ames Housing Dataset and serves predictions and analysis results as JSON endpoints, consumed by the React frontend dashboard.

## Endpoints

* `GET /` — Health check
* `GET /api/cleaning-summary` — Data cleaning decisions and row counts
* `GET /api/model-comparison` — Simple, Multiple, Ridge and Lasso model results
* `GET /api/correlation` — Top 15 feature correlations with log sale price
* `GET /api/sale-price-distribution` — Sale price histogram data
* `GET /api/neighborhood-tiers` — Median price by neighborhood tier (k-means classified)
* `GET /api/quality-vs-price` — Overall quality rating vs median sale price
* `GET /api/living-area-vs-price` — Living area vs sale price colored by neighborhood tier
* `GET /api/year-built-vs-price` — Year built vs sale price
* `POST /api/* `POST /api/* `POST /ause price from input features

## Tech Stack

* Python 3.11
* FastAPI
* Pandas
* Scikit-learn
* Uvicorn

## Setup

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Train the model: `python train_model.py`
6. Run the server: `uvicorn main:app --reload`

## Deployment

Deployed on Render.
