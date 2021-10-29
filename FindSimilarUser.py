from sklearn.preprocessing import StandardScaler
import numpy as np
from DatabaseConnector import DBConnector
from KNN import KNN_Executor


class NearestUserFinder:
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
        query_str = "select user_id, full_name, DATEDIFF(hour,u.birthday,GETDATE())/8766 AS age, CAST(u.gender AS INT) as gender,  "\
                    "common_coef, entertain_coef, gaming_coef from users u"
        df_all_users = connector.query(query_str)
        list_id_with_name = df_all_users[['user_id', 'full_name']].values.tolist()
        df_features = df_all_users.fillna(0)

        return df_features, list_id_with_name, connector

    def get_data(self):
        return self.data

    def get_query_item(self):
        # query_item_values = [30100000.0, 0.0, 12, 0.8, 0.4, 0.3, 4000, True, 1.2, 4, 128, 4.0, 1.0, 1200, 2400,
        # 4.0, 120.0, 6.2, 6.2, True, True, False, True, True, 0.16, 0.2]

        query_item_df = self.data.loc[self.data['user_id'] == self.query_id]
        query_item_df_without_id_name = query_item_df[query_item_df.columns[2:]]
        query_item_values = query_item_df_without_id_name.values.tolist()[0]
        # print("Find query item: \n", query_item_values)

        return query_item_values

    def find_nearest_neighbors(self):
        # Extract processed data
        df_without_name_and_id = self.data[self.data.columns[2:]]
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
        print("Nearest neighbors: ", k_nearest_neighbors)

        similar_users=[]
        for _, index in k_nearest_neighbors:
            # print(self.list_id_name[index])
            similar_users.append(self.list_id_name[index])

        return similar_users

    def get_high_rating_products_fromm_similar_users(self, similar_user_list, criteria_score=3.5):
        recommend_products = []
        for user_info in similar_user_list:
            user_id = user_info[0]
            query_str= "select r.product_id, p.product_name from ratings r, products p " \
                    f"where user_id='{user_id}' and score >= {criteria_score} and r.product_id = p.product_id"
            df_recommended_products = self.connector.query(query_str)
            recommend_products.extend(df_recommended_products.values.tolist())

        unique_recommend_products = list(set(tuple(sorted(sub)) for sub in recommend_products))
        # print("Recommending products from similar user: \n", recommend_products)
        return unique_recommend_products

# PD271020210047
# finder = NearestUserFinder(query_id='US281020210062', num_neighbors=7, distance_method=KNN_Executor.cal_manhattan_distance)
# print("Raw data: \n", finder.get_raw_data())

# print("Similar users:")
# similar_users = finder.find_nearest_neighbors()
# recommend_products = finder.get_high_rating_products_fromm_similar_users(similar_users)
# for product in recommend_products:
#     print(product)