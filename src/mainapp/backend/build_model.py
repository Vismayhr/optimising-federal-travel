#!/usr/bin/env python
# coding: utf-8

# In[392]:


# Imports
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pygeocoder import Geocoder
from sklearn.preprocessing import StandardScaler
from geopy.geocoders import GeocodeFarm, Nominatim
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from datetime import datetime


# In[7]:


clean_df = pd.read_csv("../data/cleaned_government_data.csv")


# In[8]:


clean_df.head(5)


# ## Origin, Destination encoding

# In[62]:


geo = Nominatim()


# In[63]:


geo_farm = GeocodeFarm()


# In[174]:


cities=list(clean_df["From"].unique())
cities.extend(list(clean_df["To"].unique()))


# In[ ]:


location_details = {}


# In[170]:


for each_city in cities:
    try:
        location = geo.geocode(each_city + ", Canada")
    except:
        try:
            location = geo_farm.geocode(each_city + ", Canada")
        except:
            continue
    try:
        location_details[each_city] = {
            "latitude": location[1][0],
            "longitude": location[1][1]
        }
    except:
        continue


# In[172]:


with open("../data/city_coordinates.json", "w") as f:
    json.dump(location_details, f)


# In[171]:


clean_df[["From_lat","From_lon"]] = clean_df["From"].apply(lambda x: pd.Series([location_details[x]["latitude"],location_details[x]["longitude"]]))
clean_df[["To_lat","To_lon"]] = clean_df["To"].apply(lambda x: pd.Series([location_details[x]["latitude"],location_details[x]["longitude"]]))


# In[178]:


clean_df.head(5)


# In[238]:


def load_coordinates():
    # Loading coords from JSON
    with open("../data/coordinates.json") as f:
        coords = json.load(f)
    return coords


# In[249]:


def transform_coordinates(clean_df):
    # Loading coordinates
    location_details = load_coordinates()
    # Origin lat, lon and destination lat, long figured
    clean_df[["From_lat","From_lon"]] = clean_df["From"].apply(lambda x: pd.Series([location_details[x]["latitude"],location_details[x]["longitude"]]))
    clean_df[["To_lat","To_lon"]] = clean_df["To"].apply(lambda x: pd.Series([location_details[x]["latitude"],location_details[x]["longitude"]]))
    clean_df.drop(["From", "To"], axis=1, inplace=True)
    return clean_df


# In[250]:


def convert_to_radians(clean_df, cols=None):
    # Converting degree to rads
    clean_df[cols] = np.radians(clean_df[cols])
    return clean_df


# In[251]:


clean_df = transform_coordinates(clean_df)
clean_df = convert_to_radians(clean_df, cols=["From_lat", "From_lon", "To_lat", "To_lon"])


# In[252]:


clean_df.head(5)


# ## Feature Engineering: Distance between origin and destination

# In[240]:


def haversine(clean_df):
    # Distance metric to calculate between two coordinate points
    from_lat, from_long, to_lat, to_long = clean_df["From_lat"], clean_df["From_lon"], clean_df["To_lat"], clean_df["To_lon"]
    radius_of_earth = 6378.1  # In km
    lat_delta = to_lat - from_lat
    lon_delta = to_long - from_long
    d = np.sin(lat_delta*0.5)**2 + np.cos(from_lat) * np.cos(to_lat) * np.sin(lon_delta*0.5)**2
    haversine_values = 2 * radius_of_earth * np.arcsin(np.sqrt(d))
    return haversine_values


# In[241]:


def calculate_distance(clean_df):
    # Add a distance column to denote distance between origin and destination
    clean_df["distance"] = clean_df.apply(haversine, axis=1)
    return clean_df


# In[212]:


clean_df = calculate_distance(clean_df)


# In[253]:


clean_df.head(5)


# ## Major Class

# In[31]:


clean_df.groupby(["Major Class"])[["Sum of Net Tickets", "Sum of Total $"]].sum()['Sum of Total $']/clean_df.groupby(["Major Class"])[["Sum of Net Tickets", "Sum of Total $"]].sum()['Sum of Net Tickets']


# ### Inference:
# - Although First Class gives a very low mean price, it is logically the most expensive way to travel via air.
# - Upon closer examination, it was seen that the first class records observed in the data was for a very short distance.
# - Thus, as per logic, it is being label encoded with First class ranking the highest and Economy ranking the lowest.
# - This might give a slightly lower accuracy for our dataset but is more interpretable.

# In[355]:


def dump_fe_pkl(model, col_name):
    with open(f"../fe_models/fe_{col_name}.pkl", "wb") as f:
        pickle.dump(model, f)


# In[356]:


def load_fe_pkl(col_name):
    with open(f"../fe_models/fe_{col_name}.pkl", "rb") as f:
        model=pickle.load(f)
    return model


# In[357]:


def label_encode(clean_df, col_name=None, use_pre_trained=False):
    if not use_pre_trained:
        model = LabelEncoder()
        model.fit(clean_df[col_name])
        dump_fe_pkl(model, col_name)
    model=load_fe_pkl(col_name)
    clean_df[col_name] = model.transform(clean_df[col_name])
    return clean_df


# In[242]:


def custom_label_encode(clean_df, mappings):
    clean_df = clean_df.replace(mappings)
    return clean_df


# In[243]:


mappings = {
    "Major Class":{
        "Economy": 1,
        "Premium Economy": 2,
        "Business Class": 3,
        "First Class": 4
    }
}
clean_df = custom_label_encode(clean_df, mappings)


# In[254]:


clean_df.head(5)


# In[271]:


def reorder_cols(clean_df, col_order=[]):
    return clean_df[col_order]


# In[272]:


clean_df.columns


# In[273]:


clean_df = reorder_cols(clean_df, col_order=['Major Class', 'Month of Travel Date', 'Sum of Net Tickets', 'From_lat', 'From_lon', 'To_lat', 'To_lon',
       'distance', 'Sum of Total $'])


# In[275]:


clean_df.head(5)


# In[375]:


def compute_avg_price(clean_df):
    clean_df["price"] = clean_df["Sum of Total $"]/clean_df["Sum of Net Tickets"]
    clean_df.drop(["Sum of Total $", "Sum of Net Tickets"], axis=1, inplace=True)
    return clean_df


# In[376]:


clean_df = compute_avg_price(clean_df)


# ## Month of Travel Date

# In[23]:


clean_df.groupby(["Month of Travel Date"])[["Sum of Net Tickets", "Sum of Total $"]].describe()


# ### Inference:
# - One-hot encode months because there is no significant rise in price between the months.
# - This was determined by diving the mean of Sum of Total with the Sum of Net Tickets for every month.


# In[422]:


def ohe_encode(clean_df, categories=None, use_pre_trained=False):
    if not use_pre_trained:
        model = make_column_transformer((categories, OneHotEncoder()), remainder="passthrough")
        clean_df_data, clean_df_labels = model.fit_transform(clean_df.iloc[:, :-1]), np.array(clean_df.iloc[:, -1])
        dump_fe_pkl(model, "ohe_model")
        return clean_df_data, clean_df_labels
    model=load_fe_pkl("ohe_model")
    clean_df = model.transform(clean_df)
    return clean_df 


# ### Also, splitting the encoded data into features and targets

# In[423]:


X, y = ohe_encode(clean_df, categories=["Month of Travel Date"])

"""
# ## Cross Validation for model selection

# In[388]:


# Use cross validation to find a model that gives the least error (using mean absolute error)

# Linear Regression on the model
lin_reg_cv_score = cross_val_score(LinearRegression(), X, y, scoring="neg_mean_absolute_error", cv=10, n_jobs=-1)
print(f"Mean absolute error with Linear Regression is: {lin_reg_cv_score}\n")

# Random Forest Regressor
forest_reg_cv_score = cross_val_score(RandomForestRegressor(), X, y, scoring="neg_mean_absolute_error", cv=10, n_jobs=-1)
print(f"Mean absolute error with Random Forest Regressor is: {forest_reg_cv_score}\n")

# XGBoost Regressor
xgb_reg_cv_score = cross_val_score(XGBRegressor(), X, y, scoring="neg_mean_absolute_error", cv=10, n_jobs=-1)
print(f"Mean absolute error with XGBoost Regressor is: {xgb_reg_cv_score}")


# ## GRIDSEARCH for Hyperparameter Tuning

# In[436]:


# Since Random Forest Regressor gives the lowest error among all models, we will use
# GridSearchCV to tune the hyper-parameters for RF Regressor and minimize the error

print(datetime.now())

rf_parameters = {'n_estimators':[120, 150, 200],
                 'max_depth':[20, 30, 50], 'min_samples_leaf':[1, 3, 5]}
rf_gsc = GridSearchCV(RandomForestRegressor(), param_grid=rf_parameters, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=True)
grid_search_result = rf_gsc.fit(X, y)
print(f"The best set of hyper-parameters are: {grid_search_result.best_params_}")

print(datetime.now())


# In[437]:


grid_search_result.estimator


# In[466]:


with open("../trained_models/model.pkl", "wb") as f:
    pickle.dump(grid_search_result.best_estimator_, f)
"""

# ## Function to preprocess data from front-end

# In[394]:


def preprocessing(clean_df):
    # Transforming "From" and "To" destinations to coordinates
    clean_df = transform_coordinates(clean_df)
    # Transforming coordinates to radians
    clean_df = convert_to_radians(clean_df, cols=["From_lat", "From_lon", "To_lat", "To_lon"])
    # Computing haversine between origin and destination
    clean_df = calculate_distance(clean_df)
    # Transforming "Major Class" using Label Encoder
    clean_df = custom_label_encode(clean_df, mappings)
    # Transforming "Month of Travel Date" using one-hot encoder
    clean_arr = ohe_encode(clean_df, categories=["Month of Travel Date"], use_pre_trained=True)
    return clean_arr


# In[493]:


sample = pd.DataFrame([["Vancouver","Halifax","Economy","Dec"]], columns=["From", "To", "Major Class", "Month of Travel Date"])


# In[494]:


sample=preprocessing(sample)


# In[495]:


sample


# In[496]:


grid_search_result.best_estimator_.predict(sample)


# In[ ]:




