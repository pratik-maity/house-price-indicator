# import joblib
# import numpy as np
# import pandas as pd

# model = joblib.load("house_price_model.pkl")
# scaler = joblib.load("scaler.pkl")
# features = joblib.load("features.pkl")

# def predict_price(input_dict):
#     # input_dict e.g. {"OverallQual":7, "GrLivArea":1500, "GarageCars":2, ...}
#     df = pd.DataFrame([input_dict])
#     # fill missing features with median 0
#     for col in features:
#         if col not in df.columns:
#             df[col] = 0
#     df = df[features] # ensure order
#     X_scaled = scaler.transform(df)
#     log_price = model.predict(X_scaled)[0]
#     price = np.expm1(log_price) # inverse log
#     return price

# if __name__ == "__main__":
#     sample = {
#         'OverallQual': 7,
#         'GrLivArea': 1710,
#         'GarageCars': 2,
#         'GarageArea': 548,
#         'TotalBsmtSF': 856,
#         '1stFlrSF': 856,
#         'FullBath': 2,
#         'YearBuilt': 2003
#     }
#     print(f"Predicted Price: ${predict_price(sample):,.2f}")



import joblib
import pandas as pd

model = joblib.load("house_price_model.pkl")

# Example house with the 8 paper features
# OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, YearBuilt
sample = pd.DataFrame([{
    'OverallQual': 7,
    'GrLivArea': 1710,
    'GarageCars': 2,
    'GarageArea': 548,
    'TotalBsmtSF': 856,
    '1stFlrSF': 856,
    'FullBath': 2,
    'YearBuilt': 2003
}])

price = model.predict(sample)[0]
print(f"Predicted House Price: ${price:,.2f}")
