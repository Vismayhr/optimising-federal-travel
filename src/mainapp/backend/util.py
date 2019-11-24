# Utilities
from backend.constants import CONSTANT
import pandas as pd
#Reads pickle into dataframe
def read_pickle(file):
    return pd.read_pickle(file)

#converts dataframe to picklefile
def pickle_dataframe(file,dataframe):
    dataframe.to_pickle(file)

# get the query number from parameters
def get_param_val(param_string):
    return CONSTANT.QUERY_OPT[param_string]
