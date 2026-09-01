# # train.py - Full pipeline replicating paper section 2.2 & 3.1
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import cross_val_score, KFold
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import mean_squared_error
# import joblib, os
# from model import get_models, get_tuning_grids
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import DecisionTreeRegressor

# # === 1. LOAD DATA ===
# # Expect Kaggle file: train.csv from https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
# # Place train.csv in same folder as this script
# CSV_PATH = "train.csv"
# if not os.path.exists(CSV_PATH):
#     print(f"ERROR: {CSV_PATH} not found. Download from Kaggle and put it here.")
#     print("Creating dummy structure so code is visible...")
#     # create dummy df for demo to run without error
#     df = pd.DataFrame({
#         'OverallQual': np.random.randint(1,10,100),
#         'GrLivArea': np.random.randint(500,4000,100),
#         'GarageCars': np.random.randint(0,4,100),
#         'GarageArea': np.random.randint(0,1000,100),
#         'FullBath': np.random.randint(0,3,100),
#         'YearBuilt': np.random.randint(1900,2020,100),
#         'TotalBsmtSF': np.random.randint(0,2000,100),
#         '1stFlrSF': np.random.randint(500,3000,100),
#         'SalePrice': np.random.randint(50000,500000,100)
#     })
# else:
#     df = pd.read_csv(CSV_PATH)
#     print(f"Loaded {df.shape}")

# # === 2. PREPROCESSING as per paper 2.2 ===
# # Target log transform (paper uses density of log price in Fig1)
# y = np.log1p(df['SalePrice']) if 'SalePrice' in df.columns else np.log1p(df.iloc[:,-1])
# X = df.drop(columns=['SalePrice','Id'], errors='ignore')

# # Handle missing + encode
# for col in X.columns:
#     if X[col].dtype == 'object':
#         X[col] = X[col].fillna('None')
#         le = LabelEncoder()
#         X[col] = le.fit_transform(X[col].astype(str))
#     else:
#         X[col] = X[col].fillna(X[col].median())

# # Simple outlier handling: cap GrLivArea > 4000 as paper mentions box-plot
# if 'GrLivArea' in X.columns:
#     X['GrLivArea'] = X['GrLivArea'].clip(upper=4000)

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Save scaler
# joblib.dump(scaler, "scaler.pkl")
# joblib.dump(list(X.columns), "features.pkl")

# # === 3. TUNING as per Fig 3,4,5 ===
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# grids = get_tuning_grids()

# print("\n--- Tuning Ridge alpha (Fig 3) ---")
# ridge_errors = []
# for alpha in grids["Ridge_alpha"]:
#     model = Ridge(alpha=alpha)
#     scores = -cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
#     ridge_errors.append(scores.mean())
#     print(f"alpha={alpha}: MSE={scores.mean():.5f}")
# best_alpha = grids["Ridge_alpha"][np.argmin(ridge_errors)]
# print(f"Best alpha: {best_alpha}")

# print("\n--- Tuning RF n_estimators (Fig 4) ---")
# rf_errors = []
# for n in grids["RF_n_estimators"]:
#     model = RandomForestRegressor(n_estimators=n, max_features=0.3, random_state=42, n_jobs=-1)
#     scores = -cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
#     rf_errors.append(scores.mean())
#     print(f"n_est={n}: MSE={scores.mean():.5f}")
# best_n = grids["RF_n_estimators"][np.argmin(rf_errors)]

# print("\n--- Tuning DT max_depth (Fig 5) ---")
# dt_errors = []
# for d in grids["DT_max_depth"]:
#     model = DecisionTreeRegressor(max_depth=d, random_state=42)
#     scores = -cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
#     dt_errors.append(scores.mean())
#     print(f"depth={d}: MSE={scores.mean():.5f}")

# # === 4. FINAL TRAIN & SAVE (Paper Table 1 result) ===
# models = {
#     "Ridge": Ridge(alpha=10),
#     "RandomForest": RandomForestRegressor(n_estimators=150, max_features=0.3, random_state=42, n_jobs=-1),
#     "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=42)
# }

# print("\n--- Final Evaluation (Table 1 replication) ---")
# for name, model in models.items():
#     scores = -cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
#     print(f"{name}: Mean MSE = {scores.mean():.5f} (paper: Ridge 0.14, RF 0.1372)")

# # Train best model (Random Forest as per paper)
# final_model = models["RandomForest"]
# final_model.fit(X_scaled, y)
# joblib.dump(final_model, "house_price_model.pkl")
# print("\nSaved: house_price_model.pkl, scaler.pkl, features.pkl")

# # Optional plots
# try:
#     plt.figure()
#     plt.plot(grids["Ridge_alpha"], ridge_errors, marker='o')
#     plt.title("Fig 3: Ridge alpha vs CV Error")
#     plt.xlabel("alpha"); plt.ylabel("MSE"); plt.savefig("fig3_ridge.png")
#     plt.figure()
#     plt.plot(grids["RF_n_estimators"], rf_errors, marker='o')
#     plt.title("Fig 4: RF n_estimators vs CV Error")
#     plt.xlabel("n_estimators"); plt.ylabel("MSE"); plt.savefig("fig4_rf.png")
#     plt.figure()
#     depths = [str(x) for x in grids["DT_max_depth"]]
#     plt.plot(depths, dt_errors, marker='o')
#     plt.title("Fig 5: DT MaxDepth vs CV Error")
#     plt.xlabel("max_depth"); plt.ylabel("MSE"); plt.savefig("fig5_dt.png")
#     print("Saved fig3, fig4, fig5")
# except Exception as e:
#     print(e)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from model import get_models, get_tuning_grids

# 1. Load
df = pd.read_csv("train.csv")
print(f"Loaded {df.shape}")

# Paper's 8 features + target
FEATURES = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt']
TARGET = 'SalePrice'

X = df[FEATURES].copy()
y = df[TARGET].copy()

# 2. Fix: fill NaN only for numeric columns (pandas 3.0 safe)
for col in X.columns:
    if X[col].isna().any():
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = get_models()
grids = get_tuning_grids()

best_score = float('inf')
best_model = None
best_name = ""

for name in models:
    print(f"\nTraining {name}...")
    grid = GridSearchCV(models[name], grids[name], cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    pred = grid.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    print(f"{name} Best Params: {grid.best_params_} | RMSE: {rmse:.2f}")

    if rmse < best_score:
        best_score = rmse
        best_model = grid.best_estimator_
        best_name = name

print(f"\nWinner: {best_name} with RMSE {best_score:.2f}")
joblib.dump(best_model, "house_price_model.pkl")
print("Saved: house_price_model.pkl")
