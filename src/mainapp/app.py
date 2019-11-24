from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
import os
import json
from backend.util import read_data
from backend.constants import CONSTANT

app = Flask(__name__, static_url_path='')
CORS(app)

latlng = None
# Load the datasets before the very first user request and have it available during the entire lifespan of the application.
# Hence, time taken for file I/O is reduced as the csv files (i.e datasets) are only read once and not for every user request.
@app.before_first_request
def load_datasets():
	print(f"Loading Global Variables....", flush=True)
	global latlng
	with open(CONSTANT.LATLNG_FILE) as latlng_file:
		latlng = json.load(latlng_file)
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



# Set host to 0.0.0.0 so that it is accessible from 'outside the container'
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 8001)))
