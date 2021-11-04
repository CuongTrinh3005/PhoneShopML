from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from DatabaseConnector import DBConnector
from KNN import KNN_Executor


class NearestNeighborsFinder:
    def __init__(self, query_id, num_neighbors, distance_method):
        self.query_id = query_id
        self.num_neighbors = num_neighbors
        self.distance_method = distance_method
        self.data, self.list_id_name, self.connector = self.get_raw_data()

    def get_raw_data(self, driver="SQL Server", servername="QUOC-CUONG", username="sa", password="cuong300599",
                     db_name="OnlinePhoneShopJoin"):
        str_for_connection = "Driver={driver};Server={servername};UID={username};PWD={password};Database={db_name};" \
            .format(driver=driver, servername=servername, username=username, password=password, db_name=db_name)

        connector = DBConnector(servername, username, password, db_name, str_for_connection)
        query_str = "SELECT * FROM dbo.all_products"
        df_all_products = connector.query(query_str)
        list_id_with_name = df_all_products[['product_id', 'product_name']].values.tolist()

        exclude_cols = ['quantity', 'description', 'specification', 'image', 'available', 'special', 'view_count', 'brand_id',
                        'category_id', 'manufacturer_id', 'common_coef', 'entertain_coef', 'gaming_coef', 'unit_price',
                        'created_date', 'updated_date', 'imei_no', 'model']
        df_features = df_all_products.drop(exclude_cols, axis=1)
        df_features = df_features.fillna(0)

        return df_features, list_id_with_name, connector

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

        query_item_df = self.data.loc[self.data['product_id'] == self.query_id]
        query_item_df_without_id_name = query_item_df[query_item_df.columns[2:]]
        query_item_values = query_item_df_without_id_name.values.tolist()[0]
        # print("Find query item: \n", query_item_values)

        return query_item_values

    def find_nearest_neighbors(self):
        # Preprocessed data
        self.data = self.preprocess_data(self.data)

        # Extract processed data
        df_without_name_and_id = self.data[self.data.columns[2:]]
        df_without_name_and_id.to_csv(r'D:\PhoneShopML\data\products.csv', header=True, index=False, encoding='utf-8-sig')
        data = df_without_name_and_id.values.tolist()

        # get query item
        query_item = self.get_query_item()

        # Scale features
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        query_item = np.array([query_item])
        query_item = scaler.transform(query_item)

        knn_model = KNN_Executor(data=data, query=query_item.flatten(), k=self.num_neighbors,
                                 distance_fn=self.distance_method
                                 , choice_fn=lambda x: None)
        k_nearest_neighbors, _ = knn_model.inference
        print("Nearest neighbors: ", k_nearest_neighbors, '\n')

        recommend_products=[]
        for _, index in k_nearest_neighbors:
            # print(self.list_id_name[index])
            recommend_products.append(self.list_id_name[index])

        return recommend_products

# PD271020210047
finder = NearestNeighborsFinder(query_id='PD271020210020', num_neighbors=5, distance_method=KNN_Executor.cal_manhattan_distance)
print("Recommend similar products for user:")
recommend_products = finder.find_nearest_neighbors()
for product in recommend_products:
    print(product)