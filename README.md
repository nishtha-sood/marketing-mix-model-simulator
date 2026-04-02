# Marketing Mix Modeling Simulator + Budget Optimizer

A regression-based Marketing Mix Model (MMM) with budget optimization and an interactive Streamlit dashboard.

## Features
- Built weekly sales data with trend and seasonality
- OLS regression model with log(1 + spend) to capture diminishing returns
- Channel attribution and ROI calculation
- Budget optimizer using scipy to maximize predicted sales under fixed total budget
- Interactive Streamlit dashboard with budget sliders and scenario analysis
- What-if analysis: Shift 20% budget from Social Meta to Paid Search

## Tech Stack
- Python, Pandas, Statsmodels, Scipy, Matplotlib, Streamlit

## How to Run Locally
1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac
