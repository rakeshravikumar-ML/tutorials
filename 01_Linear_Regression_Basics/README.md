# 📈 Synthetic Linear Regression (Beginner Friendly)

This mini project is a **gentle introduction to Machine Learning** using the simplest possible model:

> **Linear Regression** on **synthetic (fake but realistic) data**

The goal is *not* to be fancy.  
The goal is to clearly answer:

> “What does a basic ML model actually do, step by step?”

If you’re new to AI/ML, this is a great first notebook to read and run.

---

## 🧠 What this project shows

In the notebook we:

1. **Create synthetic data** where we *know* the true relationship  
   - Something like:  
     \[
     y = 4x + 10 + {small noise}
     \]
2. **Visualise the data** to see a clear upward trend  
3. **Train a Linear Regression model** with scikit-learn  
4. **Compare the learned line** with the true line  
5. **Measure errors** using MAE and RMSE  
6. **Plot the best-fit line** on top of the data points  

Because we designed the data to be linear, this is a “happy path” example where Linear Regression works really well.

---

## 🧩 Why synthetic data?

Real-world datasets (like energy consumption, stock prices, sensor data, etc.) are usually:

- noisy  
- non-linear  
- affected by many hidden factors  

So a simple straight line often struggles there.

Here, we **start with an easy world**:

- one input feature (`x`)
- one output (`y`)
- a clean linear pattern

This makes it much easier to understand:

- what the model learns  
- what the slope and intercept mean  
- why the error is small when the model is right

Later, you can compare this with real data and see why we need more advanced models.

---
