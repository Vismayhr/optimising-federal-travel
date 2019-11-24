class CONSTANT:
    QUERY_OPT = {
        'PEOPLE_FROM': 1,
        'PEOPLE_TO': 2,
        'ROUTE_FROM': 3,
        'ROUTE_TO': 4,
        'TOTAL_COST_TO': 5, # Query 5
        'TOTAL_PASSENGER_TO': 15,
        'TOTAL_COST_FROM': 6, # Query 6
        'TOTAL_PASSENGER_FROM': 16
    }

    MONTHS = {
        '1' : 'jan',
        '2' : 'feb',
        '3' : 'mar',
        '4' : 'apr',
        '5' : 'may',
        '6' : 'jun',
        '7' : 'jul',
        '8' : 'aug',
        '9' : 'sep',
        '10': 'oct',
        '11': 'nov',
        '12': 'dec'
    }

    VISUALIZATION_FOLDER = './mainapp/data/visualisation_query_data/query'

    IMP_CITIES = ['calgary','edmonton','fredericton','halifax','montreal','ottawa','quebec',
            'regina','saskatoon',"st john's",'thunder bay','toronto','vancouver','victoria','winnipeg','yellowknife']

    LATLNG_FILE = './mainapp/data/coordinates.json'

    PROB_CITIES = {
        'ft mcmurray':'Ft McMurray',
        "st john's" : "St John's"
    }
