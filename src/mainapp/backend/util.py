# Utilitiesprint(f"Starting the application...", flush=True)
from backend.constants import CONSTANT
import pandas as pd
import pickle
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
    if format_code == 1 or format_code == 2:
        format = ['city','passenger_count','total_expense','Business Class','Economy','First Class','Premium Economy']
    elif format_code == 3 or format_code == 4:
        format = ['city','passenger_count','total_expense','average_cost_per_passenger','Business Class','Economy','First Class','Premium Economy']
    elif format_code == 5 or format_code == 6:
        format = ['city','passenger_count','total_expense','Business Class','Economy','First Class','Premium Economy']
    elif format_code == 15 or format_code == 16:
        format = ['city','passenger_count','average_cost_per_passenger','Business Class','Economy','First Class','Premium Economy']
    return format

def getlatlng(city,latlng):
    city = city.lower()
    if city in CONSTANT.PROB_CITIES:
        city = CONSTANT.PROB_CITIES[city]
    else:
        city = city.title()
    return latlng[city]
# Format code is 1 for the first four and 2 for the last 4
def parse_dataframe(file_path,city,month,latlng, format_code = 1):
    response_dict = {}
    if city != None:
        response_dict['source'] = city
        response_dict['source_latlng'] = getlatlng(city,latlng)
    response_dict['month'] = month
    dataframe = read_pickle(file_path)
    format = get_data_format(format_code)
    print(dataframe,flush=True)
    city_list = []
    for index, row in dataframe.iterrows():
        city_data = {}
        for col in format:
            city_data[col] = row[col]
            if col == 'city':
                city_data['latlng'] = getlatlng(row[col],latlng)
        city_list.append(city_data)
    response_dict['data'] = city_list
    response_dict['extra'] = get_extra_data(format_code)
    return response_dict

def read_data(query, latlng, city = None, month = None):
    query_number = get_query_number(query)
    month = get_month_string(month)
    file_name = month + '.sav'
    if city != None:
        file_name = city + '_in_' + file_name
    folder_number = query_number % 10
    file_path = CONSTANT.VISUALIZATION_FOLDER + str(folder_number) + "/" + file_name
    return parse_dataframe(file_path,city,month, latlng, query_number)

def get_extra_data(query_number):
    if query_number == 15:
        query_number = 7
    elif query_number == 16:
        query_number = 8
    with open(CONSTANT.EXTRA_FOLDER+ str(query_number) +'.sav',"rb") as input_file:
        data = pickle.load(input_file)
        return data
