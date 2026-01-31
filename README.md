# Bayesian Optimization vs Random Search on Digits Dataset

This project compares **Bayesian Optimization** and **Random Search** for hyperparameter tuning of a simple **PyTorch MLP classifier** on the `sklearn` Digits dataset.  
It demonstrates how Bayesian Optimization can converge faster and achieve better validation accuracy compared to random search.

---

## 📌 Features
- Uses the **Digits dataset** from `sklearn.datasets`.
- Implements a **Multi-Layer Perceptron (MLP)** in PyTorch.
- Hyperparameters tuned:
  - Hidden layer dimension
  - Number of layers
  - Dropout rate
  - Learning rate
  - Batch size
- Compares:
  - **Bayesian Optimization** (`skopt.gp_minimize`)
  - **Random Search**
- Generates a **convergence plot** showing validation accuracy across iterations.

---

## ⚙️ Requirements
Install the required dependencies:

```bash
pip install numpy matplotlib scikit-learn torch scikit-optimize
