import flask
from flask import request, jsonify
from flask_cors import CORS

from DatabaseConnector import DBConnector
from KNN import KNN_Executor
from NearestNeighborsFinder import NearestNeighborsFinder
from RatingPredictionKNN import KNN_Rating_Prediction

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

@app.errorhandler(404)
def resource_not_found():
    return "Resource not found", 404

@app.route('/', methods=['GET'])
def home():
    return "Hello World! My name is Trinh Quoc Cuong!!!"

def recommend_similar_accessories(product_id, k):
    # Find if product has its own exclusive accessories
    recommend_all_accessories = []
    query_str = "select accessory_id from dbo.phone_accessories where product_id='{id}'".format(id=product_id)
    df_result = connector.query(query_str)
    if df_result.empty is False:
        list_accessory_ids = df_result.values.tolist()
        # Flatten a list with nested lists to a single list
        list_accessory_ids = sum(list_accessory_ids, [])

        for accessory_id in list_accessory_ids:
            finder = NearestNeighborsFinder(query_id=accessory_id, num_neighbors=k,
                                            distance_method=KNN_Executor.cal_hassanat_distance)
            recommend_accessories = finder.find_nearest_neighbors()
            recommend_all_accessories.extend(recommend_accessories)

    return recommend_all_accessories

@app.route('/api/similar-products', methods=['GET'])
def find_similar_products():
    query_parameters = request.args

    id = query_parameters.get('id', None)
    k = int(query_parameters.get('k', 10))

    if id is None:
        return resource_not_found()

    query_str = "SELECT product_id FROM dbo.all_products WHERE product_id='{id}'".format(id=id)
    df_result = connector.query(query_str)
    if df_result.empty:
        return resource_not_found()

    finder = NearestNeighborsFinder(query_id=id, num_neighbors=k, distance_method=KNN_Executor.cal_hassanat_distance)
    recommend_products = finder.find_nearest_neighbors()

    # Recommend related accessories
    recommend_accessories = recommend_similar_accessories(id, k)
    print("Number of recommend accessories in detail of product: ", len(recommend_accessories))
    if len(recommend_accessories) > 0:
        recommend_products.extend(recommend_accessories)

    unique_recommend_products = list(set(tuple(sorted(sub)) for sub in recommend_products))
    return jsonify(unique_recommend_products)

@app.route('/api/recommend-products/knn', methods=['GET'])
def recommend_products_for_user_with_knn():
    query_parameters = request.args
    k = int(query_parameters.get('k', 5))
    user_id = query_parameters.get('userid', None)
    if user_id is None:
        return resource_not_found()

    query_str = "SELECT user_id FROM dbo.users WHERE user_id='{id}'".format(id=user_id)
    df_result = connector.query(query_str)
    if df_result.empty:
        return resource_not_found()

    model = KNN_Rating_Prediction(num_neighbors=5, distance_method=KNN_Executor.cal_hassanat_distance)
    recommend_products = model.make_rating_prediction(user_id)
    for product in recommend_products:
        product_id, _ = product
        # Recommend related accessories
        recommend_accessories = recommend_similar_accessories(product_id, k)
        print("Number of recommend accessories of rating prediction: ", len(recommend_accessories))
        if len(recommend_accessories) > 0:
            recommend_products.extend(recommend_accessories)

    unique_recommend_products = list(set(tuple(sorted(sub)) for sub in recommend_products))
    return jsonify(unique_recommend_products)

@app.route('/api/recommend-products/based-viewing-history', methods=['GET'])
def recommend_products_for_user_base_on_history():
    query_parameters = request.args

    user_id = query_parameters.get('userid', None)
    k = int(query_parameters.get('k', 5))

    if user_id is None:
        return resource_not_found()

    query_str = "select product_id from dbo.view_histories where user_id='{id}'".format(id=user_id)
    df_product_ids = connector.query(query_str)
    if df_product_ids.empty:
        return resource_not_found()

    # Flatten a list with nested lists to a single list
    list_product_ids = sum(df_product_ids.values.tolist(), [])
    all_recommend_product = []
    for id in list_product_ids:
        finder = NearestNeighborsFinder(query_id=id, num_neighbors=k, distance_method=KNN_Executor.cal_hassanat_distance)
        recommend_products_with_specific_id = finder.find_nearest_neighbors()
        all_recommend_product.extend(recommend_products_with_specific_id)

        # Recommend for related accessories
        recommend_accessories = recommend_similar_accessories(id, k)
        print("Number of recommend accessories base on history: ", len(recommend_accessories))
        if len(recommend_accessories) > 0:
            all_recommend_product.extend(recommend_accessories)

    unique_recommend_products = list(set(tuple(sorted(sub)) for sub in all_recommend_product))
    return jsonify(unique_recommend_products)

app.run()