from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from DatabaseConnector import DBConnector
from KNN import KNN_Executor


class AverageRatingPredicter:
    def __init__(self, query_id, num_neighbors, distance_method):
        self.query_id = query_id
        self.num_neighbors = num_neighbors
        self.distance_method = distance_method
        self.data, self.list_id_name, self.connector = self.get_raw_data()
        self.vectorizer = None

    def get_raw_data(self, driver="SQL Server", servername="QUOC-CUONG", username="sa", password="cuong300599",
                     db_name="OnlinePhoneShopJoin"):
        str_for_connection = "Driver={driver};Server={servername};UID={username};PWD={password};Database={db_name};" \
            .format(driver=driver, servername=servername, username=username, password=password, db_name=db_name)

        connector = DBConnector(servername, username, password, db_name, str_for_connection)
        query_str = "SELECT * FROM dbo.average_ratings_view"
        df_all_products = connector.query(query_str)
        # list_id_with_name = df_all_products[['product_id', 'product_name']].values.tolist()

        exclude_cols = ['unit_price', 'discount', 'description', 'specification', 'image', 'available', 'special',
                        'view_count', 'brand_id',
                        'category_id', 'manufacturer_id', 'common_coef', 'entertain_coef', 'gaming_coef', 'warranty',
                        'created_date', 'updated_date', 'imei_no', 'model']
        df_features = df_all_products.drop(exclude_cols, axis=1)
        df_features = df_features.fillna(0)

        query_str = "select p.product_id, p.product_name from dbo.all_products p"

        df_all_products_features = connector.query(query_str).fillna(0)
        list_id_with_name = df_all_products_features[['product_id', 'product_name']].values.tolist()

        return df_features, list_id_with_name, connector

    def get_data(self):
        return self.data

    def get_average_rating_score(self, product_id):
        query_str = f"select average_rating from dbo.average_ratings_view where product_id='{product_id}'"
        df_product = self.connector.query(query_str)
        if df_product.empty is not True:
            score = df_product['average_rating'][0]
            return score
        else:
            print("Can not find product!")

    def get_average_score_TFIDF(self, list_str, vectorizer=None):
        """This function returns an average score for a string after converting to TF-IDF vector"""
        list_feature = ['' if x == 0 else x for x in list_str]
        vectorizer_Tf = None
        if vectorizer is None:
            vectorizer = TfidfVectorizer()
            vectorizer_Tf = vectorizer.fit_transform(list_feature)
            self.vectorizer = vectorizer
        else:
            vectorizer_Tf = vectorizer.transform(list_feature)

        dense_matrix = vectorizer_Tf.todense()

        list_avg_scores = []
        for row in dense_matrix:
            representative_vector = row.tolist()
            score = round(sum(representative_vector[0]) / len(representative_vector[0]), 3)
            list_avg_scores.append(score)

        return list_avg_scores

    def convert_to_numeric_df(self, list_feature_names, df_to_convert, vectorizer=None):
        """This function converts text fields to numerical data"""
        for feature in list_feature_names:
            # print("Feature: ", feature)
            list_str = df_to_convert[feature]
            list_avg_feature_score = self.get_average_score_TFIDF(list_str, vectorizer)
            df_to_convert[feature] = np.array(list_avg_feature_score)

        return df_to_convert

    def preprocess_data(self, df_to_convert, vectorizer=None):
        list_feature_names = ['compatible_devices', 'functions']
        return self.convert_to_numeric_df(list_feature_names, df_to_convert, vectorizer)

    def get_query_item(self):
        query_item_df = self.data.loc[self.data['product_id'] == self.query_id]
        query_item_df_without_id_name = query_item_df[query_item_df.columns[2:]]
        query_item_values = query_item_df_without_id_name.values.tolist()[0]

        return query_item_values

    def get_query_item_with_specification(self, specification_body):
        dict_spec_str = {'compatible_devices': [specification_body['compatible_devices']],
                         'functions': [specification_body['functions']]}
        df_str_spec = pd.DataFrame(dict_spec_str)

        df_processed_str_spec = self.preprocess_data(df_str_spec, self.vectorizer)
        compatible_devices_processed_value = df_processed_str_spec['compatible_devices'][0]
        functions_processed_value = df_processed_str_spec['functions'][0]

        specification_body['compatible_devices'] = compatible_devices_processed_value
        specification_body['functions'] = functions_processed_value
        list_feature_values = [v for v in specification_body.values()]
        print("Item's feature values: \n", list_feature_values)
        return list_feature_values

    def find_nearest_neighbors(self, specification_body=''):
        # Preprocessed data
        if self.query_id == '':
            self.data = self.preprocess_data(self.data, self.vectorizer)
        else:
            self.data = self.preprocess_data(self.data)

        # Extract processed data
        df_without_name_and_id = self.data[self.data.columns[2:]]
        df_without_name_and_id.to_csv(r'D:\PhoneShopML\data\average-ratings.csv', header=True, index=False,
                                      encoding='utf-8-sig')
        data = df_without_name_and_id.values.tolist()

        data_train = [];
        labels = []
        for item in data:
            data_point = item[:-1]
            label = item[-1]
            data_train.append(data_point)
            labels.append(label)

        # get query item
        if self.query_id != '':
            query_item = self.get_query_item()
        else:
            query_item = self.get_query_item_with_specification(specification_body)

        # Scale features
        scaler = StandardScaler()
        scale_data = scaler.fit_transform(data)
        scale_data = scale_data.tolist()
        for index in range(len(scale_data)):
            scale_data[index].append(labels[index])

        query_item = np.array([query_item])
        query_item = scaler.transform(query_item)

        knn_model = KNN_Executor(data=scale_data, query=query_item.flatten(), k=self.num_neighbors,
                                 distance_fn=self.distance_method
                                 , choice_fn=KNN_Executor.mean)
        k_nearest_neighbors, predicted_score = knn_model.inference
        actual_score = self.get_average_rating_score(self.query_id)
        if actual_score is not None:
            print("Nearest neighbors: ", k_nearest_neighbors, '\n')
            print("Predicted score: ", predicted_score)
            print("Actual score: ", actual_score)

        recommend_products = []
        for _, index in k_nearest_neighbors:
            # print(self.list_id_name[index])
            recommend_products.append(self.list_id_name[index])

        return predicted_score, actual_score, recommend_products

    def get_mse_on_entire_dataset(self):
        self.data = self.preprocess_data(self.data)
        df_without_name_and_id = self.data[self.data.columns[2:]]
        df_query_items = df_without_name_and_id.drop(['average_rating'], axis=1)

        data = df_without_name_and_id.values.tolist()

        data_train = [];
        labels = []
        for item in data:
            data_point = item[:-1]
            label = item[-1]
            data_train.append(data_point)
            labels.append(label)

        query_items = df_query_items.values.tolist()

        predict_scores = []
        for item in query_items:
            knn_model = KNN_Executor(data=data, query=item, k=self.num_neighbors,
                                     distance_fn=self.distance_method
                                     , choice_fn=KNN_Executor.mean)
            k_nearest_neighbors, predicted_score = knn_model.inference
            predict_scores.append(predicted_score)

        mse = mean_squared_error(labels, predict_scores)
        return mse


predicter = AverageRatingPredicter(query_id='PD151020210001', num_neighbors=11,
                                   distance_method=KNN_Executor.cal_hassanat_distance)
print("Mean square error in entire dataset: ", predicter.get_mse_on_entire_dataset())