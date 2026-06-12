# Combined Cycle Power Plant Energy Output Prediction & Regression Evaluation Matrix

An end-to-end machine learning regression pipeline built to predict the net hourly electrical energy output ($PE$) of a power plant based on ambient environmental readings. This framework trains, optimizes, and systematically benchmarks multiple classical regression models to isolate the architecture with the highest predictive precision.

## 📋 Project Overview

Predicting generation loads is critical for power grid stability and economic dispatch control. This repository serves as a model-selection testing ground for continuous variable estimation. It imports thermodynamic variables from a Combined Cycle Power Plant collected over 6 years, executes feature preprocessing, fits 5 distinct regression topologies, and evaluates them using goodness-of-fit indicators.

---

## 📊 Dataset Architecture (`Data.csv`)

The dataset features 9,568 data points collected from a power plant operating at full load. It contains 4 hourly average environmental variables and 1 target response variable:

| Feature Column | Physical Metric | Attribute Description |
| :--- | :--- | :--- |
| **`AT`** | Ambient Temperature | Measured in °C, ranging from 1.81°C to 37.11°C |
| **`V`** | Exhaust Vacuum | Measured in cm Hg, ranging from 25.36 to 81.56 cm Hg |
| **`AP`** | Ambient Pressure | Measured in mbar, ranging from 992.89 to 1033.30 mbar |
| **`RH`** | Relative Humidity | Measured in percentage (%), ranging from 25.56% to 100.16% |
| **`PE`** | **Net Hourly Electrical Energy Output** | **Target Continuous Variable:** Measured in MW (420.26–495.76 MW) |

---

## 🛠️ Data Preprocessing Pipeline

The processing pipeline guarantees clean feature matrices before exposing them to the algorithms:
1. **Matrix Segmentation:** Splits the continuous independent environmental features ($X$) away from the target energy output dependent vector ($y$).
2. **Train/Test Validation Split:** Allocates 80% of the observations to model training matrix arrays and reserves 20% for testing real-world generalization variance.
3. **Feature Standardization:** For models relying on geometric distance scales (specifically Support Vector Regression), `StandardScaler` handles scaling adjustments to keep features unbiased.

---

## 🤖 Regression Models Evaluated

The testing matrix evaluates and validates the following continuous prediction algorithms:

* **Multiple Linear Regression:** Evaluates baseline linear correlations across all environmental planes simultaneously.
* **Polynomial Regression:** Introduces non-linear interactions by adding higher-degree terms to model curvilinear trends.
* **Support Vector Regression (SVR):** Utilizes an RBF kernel boundary to fit predictions within an optimal margin of error ($Epsilon$).
* **Decision Tree Regression:** Breaks down features into local leaf groupings to assign average target weights.
* **Random Forest Regression:** An ensemble bagging architecture that averages predictions from numerous independent structural decision trees to curb overfitting.

---

## 📈 Performance Evaluation Metrics

To select the absolute best regression configuration, models are ranked according to standard statistical benchmarks:

* **R-Squared ($R^2$ Score):** Quantifies the proportion of target variance explained by the environmental inputs (closer to 1.0 is ideal).
* **Adjusted $R^2$:** Penalizes independent variables that do not contribute to prediction efficiency.

---

## 🚀 Quick Start & Installation

### Prerequisites
Ensure your local environment runs Python 3.8+ with these installed packages:
```bash
pip install numpy pandas scikit-learn
