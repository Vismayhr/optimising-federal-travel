# Utilitiesprint(f"Starting the application...", flush=True)
from backend.constants import CONSTANT
import pandas as pd
#Reads pickle into dataframe
def read_pickle(file):
    return pd.read_pickle(file)

#converts dataframe to picklefile
def pickle_dataframe(file,dataframe):
    dataframe.to_pickle(file)

# get the query number from parameters
def get_query_number(query):
    return CONSTANT.QUERY_OPT[query]

def get_month_string(month):
    return CONSTANT.MONTHS[month]

def get_data_format(format_code = 1):
    format = ['city','passenger_count','total_expense']
    return format

# Format code is 1 for the first four and 2 for the last 4
def parse_dataframe(file_path,city,month,format_code = 1):
    response_dict = {}
    if city != None:
        response_dict['source'] = city
    response_dict['month'] = month
    dataframe = read_pickle(file_path)
    format = get_data_format(format_code)
    city_list = []
    for index, row in dataframe.iterrows():
        city_data = {}
        for col in format:
            city_data[col] = row[col]
        city_list.append(city_data)
    response_dict['data'] = city_list
    return response_dict
    
def read_data(query,city = None, month = None):
    query_number = get_query_number(query)
    month = get_month_string(month)
    file_name = month + '.sav'
    if city != None:
        file_name = city + '_in_' + file_name
    folder_number = query_number % 10
    file_path = CONSTANT.VISUALIZATION_FOLDER + str(folder_number) + "/" + file_name
    return parse_dataframe(file_path,city,month)
