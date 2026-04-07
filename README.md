# IndiaMedicalNeuralNetwork

# Indian Healthcare AI System (Final Research Pipeline)

## Overview
This project builds a research-grade healthcare modeling system using India data from the World Bank.

It combines:
- Deep Learning (LSTM, Transformer)
- Classical ML (Linear Regression)
- Causal Inference (Difference-in-Differences)

---

## Data Sources
World Bank indicators:
- Population
- Life Expectancy
- Health Expenditure (% GDP)
- Physicians per 1000
- Hospital Beds per 1000
- Infant Mortality

---

## Models

### 1. Linear Regression
- Strong baseline
- Works well for small structured data

### 2. LSTM
- Learns temporal dependencies
- Uses sequence window of 8 years

### 3. Transformer
- Multi-head attention across time
- Captures global temporal relationships

---

## Causal Inference

Regression-based Difference-in-Differences:
- Estimates policy impact (post-2015)
- Provides statistical significance

---

## Evaluation

Metrics:
- Mean Squared Error (MSE)
- R² Score

Also includes:
- Model comparison
- Visualization of predictions

---

## Key Insights

- Linear models outperform deep learning on small datasets
- LSTM performs moderately
- Transformer needs more data to excel
- Causal inference shows statistically significant policy effects

---

## Output

- `final_plot.png` → prediction comparison
- Console output → metrics + causal summary

---

## Run

```bash
python main.py
