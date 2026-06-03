# Software Project Failure Risk Prediction System

A research-based predictive analytics platform built on real survey data.

Live URL Link : https://software-risk-predictor.streamlit.app/

## Data
- Real survey responses: n=85
- Augmented using SMOTE (Chawla et al., 2002) to address class imbalance
- RF Model Accuracy: 86.6% (5-fold cross-validation)

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push folder to GitHub (include the .xlsx file)
2. Go to share.streamlit.io
3. Select repo → Main file = app.py → Deploy

## Files
- app.py               ← Complete single-file application
- requirements.txt     ← Dependencies
- *.xlsx               ← Real survey dataset (required)
