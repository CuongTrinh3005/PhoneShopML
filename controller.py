import flask
from flask import request, jsonify
from flask_cors import CORS

from DatabaseConnector import DBConnector
from NearestNeighborsFinder import NearestNeighborsFinder

app = flask.Flask('API dispatcher')
CORS(app)
app.config["DEBUG"] = True

driver = "SQL Server"
servername = 'QUOC-CUONG'
username = 'sa'
password = 'cuong300599'
db_name = 'OnlinePhoneShopJoin'
str_for_connection = "Driver={driver};Server={servername};UID={username};PWD={password};Database={db_name};" \
            .format(driver=driver, servername=servername, username=username, password=password, db_name=db_name)

connector = DBConnector(servername, username, password, db_name, str_for_connection)

@app.route('/', methods=['GET'])
def home():
    return "Hello World! My name is Trinh Quoc Cuong!!!"

@app.route('/api/similar-products', methods=['GET'])
def find_similar_products():
    query_parameters = request.args

    id = query_parameters.get('id', None)
    k = int(query_parameters.get('k', 5))

    if id is None:
        return resource_not_found()

    query_str = "SELECT product_id FROM dbo.all_products WHERE product_id='{id}'".format(id=id)
    df_result = connector.query(query_str)
    if df_result.empty:
        return resource_not_found()

    finder = NearestNeighborsFinder(query_id=id, num_neighbors=k)
    recommend_products = finder.find_nearest_neighbors()
    return jsonify(recommend_products)

@app.errorhandler(404)
def resource_not_found():
    return "Resource not found", 404

app.run()