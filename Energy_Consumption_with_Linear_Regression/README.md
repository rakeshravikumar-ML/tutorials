# 🔋 Predicting Home Energy Consumption with Linear Regression

This mini-project shows how to use **Ordinary Least Squares (OLS) Linear Regression**  
to predict **home energy consumption** from three simple features:

- 🥵 `temperature` (°C)  
- 💧 `humidity` (%)  
- 🏠 `house_size` (square feet)

---

## 1. What are we trying to do?

Imagine you work for an energy company.

You want a quick way to **estimate how much energy a house will use**  
based on weather and house size.  

Linear Regression does exactly this:

> “When these inputs change a little, how does the output usually change?”

In ML language:

- **Features (X):** temperature, humidity, house_size  
- **Target (y):** energy_consumption  
- **Goal:** learn a straight-line relationship between X and y

---

## 2. How the model works (very simple version)

1. We collect data with inputs (X) and outputs (y).  
2. The algorithm draws a **guess line** through the data.  
3. It measures the **errors** (difference between prediction and actual).  
4. It keeps adjusting the line to make total error as small as possible.  

This “adjust until total error is smallest” trick is called  
**Ordinary Least Squares (OLS)**.

---

## 3. Project structure

Recommended files:

```text
energy_consumption_linear_regression.py  # main script with model + basic metrics
README.md                                # this file
notebooks/
  energy_regression_explained.ipynb      # (optional) notebook with plots
images/
  feature_vs_target.png
  actual_vs_pred.png
  residual_histogram.png
