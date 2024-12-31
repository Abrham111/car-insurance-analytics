from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def linear_regression(X_train, X_test, y_train, y_test):
  linear_model = LinearRegression()
  linear_model.fit(X_train, y_train)
  linear_predictions = linear_model.predict(X_test)
  linear_mse = mean_squared_error(y_test, linear_predictions)
  return linear_mse

def random_forest(X_train, X_test, y_train, y_test):
  rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  rf_predictions = rf_model.predict(X_test)
  rf_mse = mean_squared_error(y_test, rf_predictions)
  return rf_mse

def xgboost(X_train, X_test, y_train, y_test):
  xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
  xgb_model.fit(X_train, y_train)
  xgb_predictions = xgb_model.predict(X_test)
  xgb_mse = mean_squared_error(y_test, xgb_predictions)
  return xgb_mse
