# House Price Prediction API

FastAPI backend for the House Price Prediction case study. Trains a Lasso regression model on the Ames Housing Dataset (2,410 houses, 56 features) and serves predictions via REST API. Deployed on Render.

## Endpoints

- `GET /api/cleaning-summary` — Data cleaning steps and row counts
- `GET /api/model-comparison` — Simple, Multiple, Ridge and Lasso model results
- `GET /api/correlation` — Top 15 feature correlations with log sale price
- `GET /api/sale-price-distribution` — Sale price histogram data
- `GET /api/neighborhood-tiers` — Median price by neighborhood tier
- `GET /api/quality-vs-price` — Overall quality vs median sale price
- `GET /api/living-area-vs-price` — Living area vs sale price colored by tier
- `GET /api/year-built-vs-price` — Year built vs sale price
- `POST /api/predict` — Predict house price from input features

## Tech Stack
- Python, FastAPI, Scikit-learn, Pandas
- Deployed on Render
