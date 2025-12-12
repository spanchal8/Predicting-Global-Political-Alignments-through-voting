# UN Voting Prediction ML Project

Predicting UN member states' voting behavior on major resolutions using socioeconomic and military indicators.

## Project Overview

This project implements a supervised binary classification system to forecast UN voting outcomes based on:
- **Economic indicators**: GDP, trade openness, education, internet use, energy, health, tax (World Bank WDI)
- **Military indicators**: Defense spending (SIPRI)
- **Target variable**: Binary vote outcome (Yes=1, No/Abstain=0) on major topics (Security, Human Rights)

## Requirements

- Python 3.11+ (tested with Python 3.12.3)
- Virtual environment (recommended to avoid package conflicts)

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Run the project
python3 un_voting_ml_v2.py
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Run the project
python3 un_voting_ml_v2.py
```

## Troubleshooting

### Binary Incompatibility Error

If you encounter:
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

**Solution**: Use a virtual environment (as shown above) to avoid conflicts with system-installed packages.

### GPU Acceleration

For NVIDIA GPU acceleration with XGBoost:
1. Install CUDA toolkit (11.8+)
2. Install GPU-enabled XGBoost:
   ```bash
   pip install xgboost[gpu]
   ```

## Project Structure

```
Unovting/
├── un_voting_ml_v2.py      # Main ML pipeline script
├── requirements.txt        # Python dependencies
├── setup.sh               # Automated setup script
├── README.md              # This file
└── venv/                  # Virtual environment (created by setup)
```

## Output Files

The script generates the following visualization files:
- `01_distribution_analysis.png` - Data distribution analysis
- `02_correlation_analysis.png` - Feature correlations
- `03_temporal_trends.png` - Voting trends over time
- `04_model_comparison.png` - Model performance comparison
- `05_roc_curves.png` - ROC curves for all models
- `05_pr_curves.png` - Precision-Recall curves
- `06_confusion_matrices.png` - Confusion matrices
- `07_feature_importance.png` - Feature importance analysis
- `08_logistic_coefficients.png` - Logistic Regression coefficients

## Algorithms

The project implements 6 supervised ML algorithms:
1. **Logistic Regression** (Regularized L1/L2)
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **XGBoost Classifier** (GPU-accelerated if available)
5. **Support Vector Machine**
6. **Naive Bayes**

All models are tuned using GridSearchCV with stratified 5-fold cross-validation.

## Evaluation Metrics

- Accuracy, Precision, Recall, F1 Score
- ROC-AUC and PR-AUC
- Confusion Matrices
- Feature Importance Analysis
- Coefficient Analysis (Logistic Regression)

## Authors

- Devashish Mahurkar
- Shubh Panchal

## License

This project is for academic/research purposes.

