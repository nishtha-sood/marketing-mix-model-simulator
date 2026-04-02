import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

st.set_page_config(page_title="MMM Simulator", layout="wide")

st.title("Marketing Mix Modeling Simulator + Budget Optimizer")

@st.cache_data
def generate_data():
    np.random.seed(42)
    n_weeks = 104
    dates = pd.date_range('2023-01-01', periods=n_weeks, freq='W')
    channels = ['paid_search', 'social_meta', 'email', 'organic']
    spend_data = {}
    for ch in channels:
        mu = 9.0 if ch == 'paid_search' else 8.5 if ch == 'social_meta' else 7.0 if ch == 'email' else 6.0
        spend_data[ch] = np.random.lognormal(mean=mu, sigma=0.4, size=n_weeks)
    df = pd.DataFrame(spend_data, index=dates).reset_index()
    df = df.rename(columns={'index': 'date'})
    df['week_num'] = np.arange(n_weeks)
    df['trend'] = df['week_num'] * 8.0
    df['sin_season'] = np.sin(2 * np.pi * df['week_num'] / 52)
    df['cos_season'] = np.cos(2 * np.pi * df['week_num'] / 52)
    baseline = 5000 + df['trend'] + 150 * df['sin_season']
    true_betas = {'paid_search': 1200, 'social_meta': 900, 'email': 450, 'organic': 300}
    sat = 0.001
    for ch in channels:
        df[f'{ch}_true_contribution'] = true_betas[ch] * (1 - np.exp(-sat * df[ch]))
    noise = np.random.normal(0, 250, n_weeks)
    df['sales'] = baseline + df[[f'{ch}_true_contribution' for ch in channels]].sum(axis=1) + noise
    return df, channels

df, channels = generate_data()

for ch in channels:
    df[f'{ch}_log'] = np.log1p(df[ch])

X_vars = [f'{ch}_log' for ch in channels] + ['trend', 'sin_season', 'cos_season']
X = sm.add_constant(df[X_vars])
y = df['sales']
model = sm.OLS(y, X).fit()

st.sidebar.header("Budget Controls")
total_budget = st.sidebar.number_input("Total Weekly Budget ($)", value=40000, step=1000)

alloc = {}
for ch in channels:
    default = int(df[ch].mean() / df[channels].mean().sum() * 100)
    alloc[ch] = st.sidebar.slider(ch.replace('_', ' ').title(), 0, 100, default)

total_pct = sum(alloc.values())
user_alloc = {ch: round(alloc[ch] / total_pct * total_budget) for ch in channels}

st.subheader("ROI Summary")
contribs = {}
for var in [v for v in X_vars if '_log' in v]:
    ch = var.replace('_log', '')
    contribs[ch] = model.params[var] * df[f'{ch}_log']
roi = {ch: round(contribs[ch].sum() / df[ch].sum(), 2) if df[ch].sum() > 0 else 0 for ch in channels}
st.dataframe(pd.DataFrame(list(roi.items()), columns=["Channel", "ROI ($)"]), use_container_width=True)

st.subheader("Your Current Allocation")
st.write({ch.replace('_', ' ').title(): f"${amt:,}" for ch, amt in user_alloc.items()})

# Optimizer
def predict_sales(budgets):
    log_budgets = np.log1p(budgets)
    X_new = np.concatenate(([1.0], log_budgets, X.mean().values[-3:]))
    return np.dot(X_new, model.params.values)

def objective(budgets):
    return -predict_sales(budgets)

initial_guess = df[channels].mean().values
cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}
bounds = [(0, None) for _ in channels]

result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=cons)
optimal_alloc = dict(zip(channels, result.x.round(0)))

st.subheader("Optimizer Recommended Allocation")
st.write({ch.replace('_', ' ').title(): f"${amt:,}" for ch, amt in optimal_alloc.items()})

st.subheader("Scenario: Shift 20% from Social Meta to Paid Search")
new_spends = df[channels].copy()
amount = df['social_meta'].sum() * 0.20
new_spends['social_meta'] *= 0.80
new_spends['paid_search'] += amount / len(df)

X_scenario = X.copy()
for ch in channels:
    X_scenario[f'{ch}_log'] = np.log1p(new_spends[ch])

new_sales = model.predict(sm.add_constant(X_scenario[X_vars]))
lift = (new_sales.mean() - df['sales'].mean()) / df['sales'].mean() * 100
st.write(f"Expected sales change: {lift:+.1f}%")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['date'], model.predict(X), label='Baseline Model')
ax.plot(df['date'], new_sales, label='After 20% Shift')
ax.set_xlabel('Date')
ax.set_ylabel('Predicted Sales')
ax.legend()
st.pyplot(fig)

st.caption("Marketing Mix Model Simulator and Optimizer")
