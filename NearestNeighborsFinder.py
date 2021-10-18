from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from DatabaseConnector import DBConnector
from KNN import KNN_Executor


class NearestNeighborsFinder:
    def __init__(self, query, num_neighbors):
        self.query = query
        self.num_neighbors = num_neighbors
        self.data = self.get_data()

    @classmethod
    def get_data(cls, driver="SQL Server",servername="QUOC-CUONG", username="sa", password="cuong300599", db_name="OnlinePhoneShopJoin"):
        str_for_connection = "Driver={driver};Server={servername};UID={username};PWD={password};Database={db_name};" \
            .format(driver=driver, servername=servername, username=username, password=password, db_name=db_name)

        connector = DBConnector(servername, username, password, db_name, str_for_connection)
        query_str = "SELECT * FROM dbo.all_products"
        df_products = connector.query(query_str)
        df_products.fillna(0)
        return df_products

    def find_nearest_neighbors(self):
        knn_model = KNN_Executor(data=self.data, query=self.query, k=5, distance_fn=KNN_Executor.cal_euclidean_distance
                                 , choice_fn=None)
        k_nearest_neighbors, _ = knn_model.inference()
        print("Nearest neighbors: ", k_nearest_neighbors)

    def get_average_score_TFIDF(self, list_str):
        """This function returns an average score for a string after converting to TF-IDF vector"""
        list_feature = ['' if x == 0 else x for x in list_str]
        vectorizer = TfidfVectorizer()
        vectorizer_Tf = vectorizer.fit_transform(list_feature)
        dense_matrix = vectorizer_Tf.todense()
        print("Dense matrix: \n", dense_matrix)
        #     print("Vector to dense: ", vectorizer_Tf, " and its shape: ", vectorizer_Tf.shape)

        list_avg_scores = []
        for row in dense_matrix:
            representative_vector = row.tolist()
            score = round(sum(representative_vector[0]) / len(representative_vector[0]), 3)
            list_avg_scores.append(score)

        return list_avg_scores

    def convert_to_numeric_df(self, list_feature_names):
        """This function converts text fields to numerical data"""
        for feature in list_feature_names:
            list_str = self.data[feature]
            list_avg_feature_score = self.get_average_score_TFIDF(list_str)
            self.data[feature] = np.array(list_avg_feature_score)

data = NearestNeighborsFinder.get_data()
print("Data here: \n", data)

list_feature_names = ['compatible_devices', 'functions']
finder = NearestNeighborsFinder(query=None, num_neighbors=3)
finder.convert_to_numeric_df(list_feature_names)
finder.data.head()