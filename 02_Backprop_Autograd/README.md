# Backpropagation & Autograd in TensorFlow – Telecom Traffic Demo

This repo is a **small, production-flavoured tutorial** that shows how backpropagation and autograd work in TensorFlow using a **telecom-style traffic forecasting problem**.

Instead of a toy `y = 0.5x + 1`, we simulate hourly network load with daily and weekly seasonality, then train a regression model with a **custom training loop built on `tf.GradientTape`**.

---

## 🎯 What this project demonstrates

- Practical understanding of **backpropagation** beyond the equations  
- Using **TensorFlow 2.x autograd (`tf.GradientTape`)** explicitly  
- Building a **custom training loop** instead of relying only on `model.fit`  
- Handling data with `tf.data.Dataset` and a realistic train/validation split  
- Framing a simple but believable **telecom traffic forecasting** problem

This is meant to look like a compact, real-world ML experiment that you might run during prototyping.

---

## 📂 Repository structure

- `backprop_autograd_tensorflow_telecom.ipynb`  
  Notebook with explanations, diagrams (forward/backward graphs), and full training/evaluation runs.

- `telecom_backprop_tf.py`  
  Script version of the experiment – same logic as the notebook, easy to run from the command line.

---

## 🧱 Problem setup

We simulate 120 days of hourly telecom traffic and engineer three simple features:

- `hour_of_day` (0–23)  
- `day_of_week` (0–6)  
- `prev_hour_traffic` (lag-1 of the target)

The target is the **current hour traffic**, generated from a mix of daily and weekly sinusoidal patterns plus noise.

The model:

```text
Input(3) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1)

