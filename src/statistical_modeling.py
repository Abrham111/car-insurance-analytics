import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
# import shap

# Linear Regression function
def linear_regression(X_train, X_test, y_train, y_test):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_predictions)
    return lr_model, lr_mse

# Random Forest function
def random_forest(X_train, X_test, y_train, y_test):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_predictions)
    return rf_model, rf_mse

# XGBoost function
def xgboost_model(X_train, X_test, y_train, y_test):
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    xgb_model.fit(dtrain)
    xgb_predictions = xgb_model.predict(dtest)
    xgb_mse = mean_squared_error(y_test, xgb_predictions)
    return xgb_model, xgb_mse

# SHAP analysis for models
# def shap_analysis(model, X):
#     explainer = shap.Explainer(model)
#     shap_values = explainer(X)
    
#     # Summary plot
#     shap.summary_plot(shap_values, X)