# # model.py - defines 3 models as per paper
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import DecisionTreeRegressor

# def get_models():
#     # Paper best params
#     ridge = Ridge(alpha=10)  # paper best alpha 10-20
#     rf = RandomForestRegressor(n_estimators=150, max_features=0.3, random_state=42, n_jobs=-1)
#     dt = DecisionTreeRegressor(max_depth=10, random_state=42)
#     return {
#         "Ridge(alpha=10)": ridge,
#         "RandomForest(n=150,mf=0.3)": rf,
#         "DecisionTree(depth=10)": dt
#     }

# def get_tuning_grids():
#     return {
#         "Ridge_alpha": [1,5,10,15,20,50,100],
#         "RF_n_estimators": [10,50,100,150,200,300],
#         "DT_max_depth": [2,4,6,8,10,12,15,20,None]
#     }


from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

def get_models():
    return {
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42)
    }

def get_tuning_grids():
    return {
        "Ridge": {"alpha": [0.1, 1, 10, 100]},
        "RandomForest": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
        "DecisionTree": {"max_depth": [None, 10, 20, 30]}
    }
