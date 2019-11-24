from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
import os
import json, pickle
import numpy as np
from backend.util import read_data
from backend.constants import CONSTANT
from backend.build_model import preprocessing, make_df

app = Flask(__name__, static_url_path='')
CORS(app)

latlng = None
passenger_trends = None
model = None
# Load the datasets before the very first user request and have it available during the entire lifespan of the application.
# Hence, time taken for file I/O is reduced as the csv files (i.e datasets) are only read once and not for every user request.
@app.before_first_request
def load_datasets():
	print(f"Loading Global Variables....", flush=True)
	global latlng
	global passenger_trends
	global model
	with open(CONSTANT.LATLNG_FILE) as latlng_file:
		latlng = json.load(latlng_file)
	with open(CONSTANT.PREDICTION_PAGE_QUERIES, "rb") as pass_count_trend:
		passenger_trends = pickle.load(pass_count_trend)
	with open(CONSTANT.MODEL_PATH, "rb") as model_obj:
		model = pickle.load(model_obj)
	print(f"Loading Successful....", flush=True)


@app.route('/init', methods=['GET'])
def init():
    # "test method"
    return "Init method called"


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(error):
    return render_template('500.html'), 500

#No Home path start with the map path
@app.route('/')
def go_home():
	return redirect('map', code=303)

@app.route('/map')
def base_map():
	query = request.args.get('query')
	city = request.args.get('city')
	month = request.args.get('month')
	return read_data(query, latlng, city, month) #render_template('index.html')

@app.route('/cities')
def imp_cities():
	return json.dumps(CONSTANT.IMP_CITIES)

@app.route('/get-predictions')
def get_predictions():
	origin = request.args.get('origin')
	destination = request.args.get('destination')
	month = request.args.get('month')
	number = int(request.args.get('number'))
	majorclass = request.args.get('majorclass')
	index = CONSTANT.MONTH_LIST.index(month)
	index = index-12 if index>9 else index
	months = [CONSTANT.MONTH_LIST[index+i] for i in range(-2, 3)]
	X_test_df = make_df(origin, destination, majorclass, months)
	X_test = preprocessing(X_test_df)
	preds_arr = model.predict(X_test)
	preds_arr = preds_arr * number
	costs = dict(zip(months, list(preds_arr)))
	lowest_month_index = np.argmin(preds_arr)
	highest_month_index = np.argmax(preds_arr)
	return {
		"preds":{
			"costs": costs,
			"lowest_cost_month": months[lowest_month_index],
			"highest_cost_month": months[highest_month_index]
		},
		"passenger_trends": passenger_trends,
		"optimal_time": CONSTANT.OPTIMAL_DATE[CONSTANT.MONTH_LIST.index(month)]
	}
# Set host to 0.0.0.0 so that it is accessible from 'outside the container'
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 8001)))
