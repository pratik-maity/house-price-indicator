# 🏠 House Price Predictor — From Paper to Real-World ML App

> A complete end-to-end Machine Learning project that predicts house prices based on 8 key features, trained on the Ames Housing dataset (Kaggle). Built as a research paper implementation and deployed as an interactive web app.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![Streamlit](https://img.shields.io/badge/Deployed%20on-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

**Live Demo:** https://house-price-indicator.streamlit.app/

### 📸 Preview
Your web app looks like this:
- User enters 8 features via sliders & inputs
- Clicks Predict -> Gets estimated price instantly with balloons 🎈
- RMSE ~ $29,760

---

### 📄 Paper to Project — What I Implemented

Original paper proposed house price prediction using multiple regression models.
This implementation follows:

1.  **Dataset:** `train.csv` from Kaggle House Prices: Advanced Regression Techniques (1460 rows x 81 cols)
2.  **Feature Selection:** Selected top 8 most correlated features based on domain knowledge & correlation analysis:
    - `OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`, `FullBath`, `YearBuilt`
3.  **Models Tried:**
    - Ridge Regression (with GridSearchCV)
    - Lasso Regression (with GridSearchCV)
    - RandomForest Regressor (Winner — lowest RMSE)
4.  **Training:** 5-Fold Cross Validation, GridSearch for best hyperparams
5.  **Evaluation Metric:** RMSE (Root Mean Squared Error)
6.  **Deployment:** Streamlit webapp

### 🧠 Model Performance

| Model | Best Params | RMSE |
|-------|-------------|------|
| Ridge | alpha=10 | ~32,000 |
| Lasso | alpha=1000 | ~31,500 |
| **RandomForest** | n_estimators=200 | **~29,760** |

**Winner:** RandomForest saved as `house_price_model.pkl`

### 📂 Project Structure

```
house-price-indicator/
│
├── app.py                      # Streamlit Web App (for deployment)
├── train.py                    # Trains all models + saves best
├── model.py                    # Model definitions
├── predict.py                  # CLI prediction for testing
├── house_price_model.pkl       # Trained model (generated)
├── train.csv                   # Dataset (not needed for deployment)
├── requirements.txt            # Dependencies
└── README.md
```

### 🚀 How to Run Locally

**1. Clone / Download**
```bash
git clone https://github.com/yourusername/house-price-predictor.git
cd house-price-predictor
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Train (Optional — already have .pkl)**
```bash
python train.py
```

**4. Test in Terminal**
```bash
python predict.py
# Output: Predicted House Price: $200,799.56
```

**5. Run Web App**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

### 🌐 Deploy for FREE — Streamlit Cloud

1. Push `app.py`, `house_price_model.pkl`, `requirements.txt`, `README.md` to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub > New App > Select Repo > File: `app.py` > Deploy
4. You get a public shareable link

### 🛠️ Tech Stack

- Python 3.12
- pandas, numpy
- scikit-learn (Ridge, Lasso, RandomForest, GridSearchCV)
- matplotlib, seaborn (for EDA in train.py)
- joblib (model saving)
- Streamlit (deployment)

### 📦 Requirements

```
pandas
scikit-learn
joblib
streamlit
```

### 🎯 Future Improvements

- [ ] Add more features (Neighborhood, LotArea)
- [ ] Try XGBoost / LightGBM
- [ ] Add SHAP explainability
- [ ] Dockerize the app

### 👨‍💻 Author

**Pratik Maity** 
<!--
- Built as part of ML learning journey
- From `KeyError` to deployed app
-->

### 📝 License

MIT — Free to use, modify, distribute.

---
⭐ If you liked this, give it a star on GitHub!
