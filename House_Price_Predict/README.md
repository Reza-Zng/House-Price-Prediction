# 🏢 Tehran House Price Prediction & Web App

An end-to-end Machine Learning project to predict residential property prices in Tehran (in USD). It features a complete data pipeline—from exploratory analysis to hyperparameter tuning—and includes an interactive **Streamlit web interface** for inference.

---

## 📌 Project Overview
* **Dataset:** Tehran House Price Dataset (Spring 1400 / 2021, crawled from Divar.ir by Soheil Tehranipour).
* **Goal:** Estimate property prices based on area, number of rooms, location (neighborhood), and key amenities.
* **Best Model:** **Extra Trees Regressor** tuned via 5-Fold Grid Search Cross-Validation.
* **Deployment:** Interactive Web App built with **Streamlit**.

---

## 🛠️ Pipeline & Features

1. **Data Cleaning & Preprocessing:**
   * Handled string formatting and numeric conversions for `Area`.
   * Standardized boolean flags for `Parking`, `Warehouse`, and `Elevator`.
   * Applied domain-knowledge thresholding to filter domain-specific outliers.
2. **Exploratory Data Analysis (EDA):**
   * Target distribution analysis (`Price(USD)`).
   * Feature correlation heatmaps and amenity distributions.
   * Visualizing price dynamics across different Tehran neighborhoods.
3. **Model Evaluation & Comparison:**
   * Evaluated multiple algorithms: `Random Forest`, `Gradient Boosting`, `XGBoost`, `Extra Trees`, `KNN`, and `Bagging`.
   * Cross-validation ($R^2$, RMSE, MAE) and overfitting analysis ($CV\ R^2$ vs $Test\ R^2$).
   * Selected **Extra Trees Regressor** as the top performer.
4. **Interactive Deployment:**
   * Web interface built using Streamlit with custom CSS layout.
   * Real-time house price estimation using stored artifact models (`.pkl`).

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
git clone [https://github.com/your-username/tehran-house-price-prediction.git](https://github.com/your-username/tehran-house-price-prediction.git)
cd tehran-house-price-prediction
pip install -r requirements.txt