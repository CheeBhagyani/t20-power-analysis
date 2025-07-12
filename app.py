import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.power import TTestIndPower

# --- Sidebar: User Inputs ---
st.sidebar.header("Batting Pair Parameters")

mu_A = st.sidebar.number_input("Mean of Pair A (μ₁)", min_value=0.0, value=8.5, step=0.1)
mu_B = st.sidebar.number_input("Mean of Pair B (μ₂)", min_value=0.0, value=9.2, step=0.1)
sigma = st.sidebar.number_input("Standard Deviation (σ)", min_value=0.1, value=1.5, step=0.1)
alpha = st.sidebar.slider("Significance Level (α)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
max_sample_size = st.sidebar.slider("Max Sample Size per Group", min_value=20, max_value=200, value=100, step=10)
n_simulations = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)

# --- Core Logic ---
effect_size = (mu_B - mu_A) / sigma
sample_sizes = range(5, max_sample_size + 1, 5)

# Analytical power calculation
def analytical_power_curve(effect_size, alpha, sample_sizes):
    power_analysis = TTestIndPower()
    return [power_analysis.power(effect_size=effect_size, nobs1=n, alpha=alpha, ratio=1.0) for n in sample_sizes]

# Simulation-based power estimation
def simulate_power(mu_A, mu_B, sigma, alpha, sample_size, n_simulations):
    rejections = 0
    for _ in range(n_simulations):
        A = np.random.normal(mu_A, sigma, sample_size)
        B = np.random.normal(mu_B, sigma, sample_size)
        _, p = ttest_ind(A, B)
        if p < alpha:
            rejections += 1
    return rejections / n_simulations

def simulated_power_curve(mu_A, mu_B, sigma, alpha, sample_sizes, n_simulations):
    return [simulate_power(mu_A, mu_B, sigma, alpha, n, n_simulations) for n in sample_sizes]

# Calculate power
st.write("### Power Curve for T20 Batting Pair Comparison")
with st.spinner("Calculating power curves..."):
    analytical_powers = analytical_power_curve(effect_size, alpha, sample_sizes)
    simulated_powers = simulated_power_curve(mu_A, mu_B, sigma, alpha, sample_sizes, n_simulations)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sample_sizes, analytical_powers, label='Analytical Power', color='blue', marker='o')
ax.plot(sample_sizes, simulated_powers, label='Simulated Power', color='green', marker='x')
ax.axhline(0.8, linestyle='--', color='red', label='Target Power = 0.8')
ax.set_xlabel("Sample Size per Group")
ax.set_ylabel("Power")
ax.set_title("Power Curve: Batting Pair Comparison")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# --- Summary ---
mean_diff = mu_B - mu_A
st.write(f"**Mean Difference (μ₂ - μ₁):** {mean_diff:.2f} runs/over")
st.write(f"**Effect Size (Cohen's d):** {effect_size:.2f}")
st.info("A power of 0.8 or more is typically considered sufficient to detect a real difference.")
