from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from DatabaseConnector import DBConnector
from KNN import KNN_Executor


class AverageRatingPredicter:
    def __init__(self, query_id, num_neighbors, distance_method):
        self.query_id = query_id
        self.num_neighbors = num_neighbors
        self.distance_method = distance_method
        self.data, self.connector = self.get_raw_data()
        self.list_id_name = []

    def get_raw_data(self, driver="SQL Server", servername="QUOC-CUONG", username="sa", password="cuong300599",
                     db_name="OnlinePhoneShopJoin"):
        str_for_connection = "Driver={driver};Server={servername};UID={username};PWD={password};Database={db_name};" \
            .format(driver=driver, servername=servername, username=username, password=password, db_name=db_name)

        connector = DBConnector(servername, username, password, db_name, str_for_connection)
        query_str = "SELECT * FROM dbo.average_ratings_view"
        df_all_products = connector.query(query_str)
        # list_id_with_name = df_all_products[['product_id', 'product_name']].values.tolist()

        exclude_cols = ['unit_price', 'discount', 'description', 'specification',  'image', 'available', 'special', 'view_count', 'brand_id',
                        'category_id', 'manufacturer_id', 'common_coef', 'entertain_coef', 'gaming_coef',
                        'created_date', 'updated_date', 'imei_no', 'model']
        df_features = df_all_products.drop(exclude_cols, axis=1)
        df_features = df_features.fillna(0)

        return df_features, connector

    def get_data(self):
        return self.data

    def get_average_score_TFIDF(self, list_str):
        """This function returns an average score for a string after converting to TF-IDF vector"""
        list_feature = ['' if x == 0 else x for x in list_str]
        vectorizer = TfidfVectorizer()
        vectorizer_Tf = vectorizer.fit_transform(list_feature)
        dense_matrix = vectorizer_Tf.todense()
        # print("Dense matrix: \n", dense_matrix, '\n')

        list_avg_scores = []
        for row in dense_matrix:
            representative_vector = row.tolist()
            score = round(sum(representative_vector[0]) / len(representative_vector[0]), 3)
            list_avg_scores.append(score)

        return list_avg_scores

    def convert_to_numeric_df(self, list_feature_names, df_to_convert):
        """This function converts text fields to numerical data"""
        for feature in list_feature_names:
            # print("Feature: ", feature)
            list_str = df_to_convert[feature]
            list_avg_feature_score = self.get_average_score_TFIDF(list_str)
            df_to_convert[feature] = np.array(list_avg_feature_score)

        return df_to_convert

    def preprocess_data(self, df_to_convert):
        list_feature_names = ['compatible_devices', 'functions']
        return self.convert_to_numeric_df(list_feature_names, df_to_convert)

    def get_query_item(self):
        # query_item_values = [30100000.0, 0.0, 12, 0.8, 0.4, 0.3, 4000, True, 1.2, 4, 128, 4.0, 1.0, 1200, 2400,
        # 4.0, 120.0, 6.2, 6.2, True, True, False, True, True, 0.16, 0.2]

        # query_item_df = self.data.loc[self.data['product_id'] == self.query_id]
        # query_item_df_without_id_name = query_item_df[query_item_df.columns[2:]]
        # query_item_values = query_item_df_without_id_name.values.tolist()[0]
        # # print("Find query item: \n", query_item_values)

        query_str = "select p.product_id, p.product_name, p.ram, p.rom, p.battery_power, p.resolution, p.max_core," \
                    "p.max_speed, p.refresh_rate, p.sim_support, p.networks, p.no_front_cam, p.touch_screen, p.wifi, p.bluetooth," \
                    "p.compatible_devices, p.functions, p.label, p.warranty " \
                    "from dbo.all_products p"
        df_all_products_features = self.connector.query(query_str).fillna(0)
        self.list_id_name = df_all_products_features[['product_id', 'product_name']].values.tolist()
        df_all_products_features = df_all_products_features.drop(['product_id', 'product_name'], axis=1)
        df_all_products_features = self.preprocess_data(df_all_products_features)

        return df_all_products_features.values.tolist()[0]

    def get_average_rating_score(self, product_id):
        query_str = f"select average_rating from dbo.average_ratings_view where product_id='{product_id}'"
        df_product = self.connector.query(query_str)
        if df_product.empty is not True:
            score = df_product['average_rating'][0]
            return score
        else:
            print("Can not find product!")

    def find_nearest_neighbors(self):
        # Preprocessed data
        self.data = self.preprocess_data(self.data)

        # Extract processed data
        df_without_name_and_id = self.data[self.data.columns[2:]]
        data = df_without_name_and_id.values.tolist()

        # get query item
        query_item = self.get_query_item()

        # Scale features
        # scaler = StandardScaler()
        # data = scaler.fit_transform(data)
        #
        # query_item = np.array([query_item])
        # query_item = scaler.transform(query_item)

        knn_model = KNN_Executor(data=data, query=query_item, k=self.num_neighbors,
                                 distance_fn=self.distance_method
                                 , choice_fn=KNN_Executor.mean)
        k_nearest_neighbors, predicted_score = knn_model.inference
        actual_score = self.get_average_rating_score(self.query_id)
        if actual_score is not None:
            print("Nearest neighbors: ", k_nearest_neighbors, '\n')
            print("Predicted score: ", predicted_score)
            print("Actual score: ", actual_score)

        recommend_products=[]
        for _, index in k_nearest_neighbors:
            # print(self.list_id_name[index])
            recommend_products.append(self.list_id_name[index])

        return predicted_score, actual_score, recommend_products

predicter = AverageRatingPredicter(query_id='PD041020210005', num_neighbors=9, distance_method=KNN_Executor.cal_euclidean_distance)
# print("Raw data: \n", finder.get_raw_data())
# print("Get query item: \n")
# query_items = finder.get_query_item()
# for item in query_items:
#     print(item)

# print("Recommend similar products for user:\n")
recommend_products = predicter.find_nearest_neighbors()
for product in recommend_products:
    print(product)