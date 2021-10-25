import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from DatabaseConnector import DBConnector
from KNN import KNN_Executor


class KNN_Rating_Prediction:
    def __init__(self, num_neighbors, distance_method):
        self.num_neighbors = num_neighbors
        self.distance_method = distance_method
        self.data, self.connector, self.mean_age = self.get_raw_data()
        self.list_id_name = []

    def get_raw_data(self, driver="SQL Server", servername="QUOC-CUONG", username="sa", password="cuong300599",
                     db_name="OnlinePhoneShopJoin"):
        str_for_connection = "Driver={driver};Server={servername};UID={username};PWD={password};Database={db_name};" \
            .format(driver=driver, servername=servername, username=username, password=password, db_name=db_name)

        connector = DBConnector(servername, username, password, db_name, str_for_connection)
        query_str = "SELECT * FROM dbo.users_ratings_products"
        df_users_ratings_products = connector.query(query_str)

        exclude_cols = ['product_id', 'product_name', 'gender']
        df_features = df_users_ratings_products.drop(exclude_cols, axis=1)

        # Replace nan in age with mean columns
        mean_age = int(round(df_features['age'].mean()))
        df_features['age'].fillna((mean_age), inplace=True)
        df_features = df_features.fillna(0)

        return df_features, connector, mean_age

    def get_data(self):
        return self.data

    def get_average_score_TFIDF(self, list_str):
        """This function returns an average score for a string after converting to TF-IDF vector"""
        list_feature = ['' if x == 0 else x for x in list_str]
        vectorizer = TfidfVectorizer()
        vectorizer_Tf = vectorizer.fit_transform(list_feature)
        dense_matrix = vectorizer_Tf.todense()

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

    def get_query_items(self, userId):
        # Get query user
        query_str = "select DATEDIFF(hour,u.birthday,GETDATE())/8766 AS age, u.common_coef, u.entertain_coef, u.gaming_coef from users u " \
                    f"where u.user_id='{userId}'"
        df_user_query = self.connector.query(query_str).fillna(0)
        user_age, user_common_demand, user_entertainment_demand, user_gaming_demand = df_user_query.values.tolist()[0]
        if user_age == 0:
            user_age = self.mean_age

        # Get all product's features
        query_str = "select p.product_id, p.product_name, p.unit_price, p.discount, p.battery_power, p.bluetooth, p.clock_speed, p.front_cam, p.in_memory, " \
                    "p.ram, p.refresh_rate, p.n_cores, p.n_sim, p.px_height, p.px_width, p.screen_height, p.screen_width, p.touch_screen, p.wifi, p.support_3g, p.support_4g," \
                    "p.support_5g, p.warranty, p.compatible_devices, p.functions, p.type	" \
                    "from dbo.all_products p"
        df_all_products_features = self.connector.query(query_str).fillna(0)
        self.list_id_name = df_all_products_features[['product_id', 'product_name']].values.tolist()
        df_all_products_features = df_all_products_features.drop(['product_id', 'product_name'], axis=1)
        df_all_products_features = self.preprocess_data(df_all_products_features)

        # Combine user's features with all products' feature
        df_user_rating_products = df_all_products_features
        df_user_rating_products['common_demand'] = [user_common_demand] * len(df_user_rating_products)
        df_user_rating_products['entertainment_demand'] = [user_entertainment_demand] * len(df_user_rating_products)
        df_user_rating_products['gaming_demand'] = [user_gaming_demand] * len(df_user_rating_products)
        df_user_rating_products['age'] = [user_age] * len(df_user_rating_products)

        first_column = df_user_rating_products.pop('age')
        df_user_rating_products.insert(0, 'age', first_column)

        second_column = df_user_rating_products.pop('common_demand')
        df_user_rating_products.insert(1, 'common_demand', second_column)
        third_column = df_user_rating_products.pop('entertainment_demand')
        df_user_rating_products.insert(2, 'entertainment_demand', third_column)
        fourth_column = df_user_rating_products.pop('gaming_demand')
        df_user_rating_products.insert(3, 'gaming_demand', fourth_column)

        return df_user_rating_products.values.tolist()

    def make_rating_prediction(self, userId, rating_criteria_score=3.5):
        # Preprocessed data
        self.data = self.preprocess_data(self.data)

        # Extract processed data
        df_without_name_and_id = self.data[self.data.columns[2:]]
        # Save to csv
        df_without_name_and_id.to_csv(r'D:\PhoneShopML\user-rating.csv', index=False)

        # print("Data: \n", df_without_name_and_id)
        data = df_without_name_and_id.values.tolist()

        # get query item
        # query_item = [0, 0, 0, 2000000, 0, 3400, True, 1.2, 4, 32, 2, 120, 2, 1, 720, 1600, 6.4, 6.4, True, True, True, True, False, 12, 1
        #     , 0.7, 0.6, 0.4, 0.16, 0.2]
        query_items = self.get_query_items(userId)

        # scaler = StandardScaler()
        # data_to_scale = []
        # for data_item in data:
        #     data_item = data_item[:-1]
        #     data_to_scale.append(data_item)
        #
        # data_to_scale = scaler.fit_transform(data_to_scale)
        recommend_products = []
        for index, item in enumerate(query_items):
            # print('Query length: ', len(item))
            # item = np.array([item])
            # item = scaler.transform(item)
            knn_model = KNN_Executor(data=data, query=item, k=self.num_neighbors,
                                     distance_fn=self.distance_method
                                     , choice_fn=KNN_Executor.mean, match_exactly=True)
            k_nearest_neighbors, rating_predictions = knn_model.inference
            # Round off rating to nearest 0.5. Ex: 2.3 -> 2.5
            rating_round_off = round(rating_predictions * 2) / 2
            if rating_round_off >= rating_criteria_score:
                print("Rating predictions: ", rating_round_off)
                item_to_recommend = self.list_id_name[index]
                print(f"Product info: {item_to_recommend}")
                recommend_products.append(item_to_recommend)

        return recommend_products

    def get_accuracy_with_data(self):
        # Preprocessed data
        self.data = self.preprocess_data(self.data)

        # Extract processed data
        df_without_name_and_id = self.data[self.data.columns[2:]]

        # Extracting Independent and dependent Variable
        x = df_without_name_and_id.iloc[:, :-1].values
        y = df_without_name_and_id.iloc[:, -1].values

        # Use label encoder
        from sklearn import preprocessing
        lab_enc = preprocessing.LabelEncoder()
        encoded_label = lab_enc.fit_transform(y)

        # Splitting the dataset into training and test set.
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, encoded_label, test_size=0.1, random_state=0)

        # feature Scaling
        from sklearn.preprocessing import StandardScaler
        st_x = StandardScaler()
        x_train = st_x.fit_transform(x_train)
        x_test = st_x.transform(x_test)

        # Fitting K-NN classifier to the training set
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        classifier.fit(x_train, y_train)

        # Predicting the test set result
        y_pred = classifier.predict(x_test)

        # Creating the Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion matrix:\n", cm)

        # Get the accuracy
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_test, y_pred)


model = KNN_Rating_Prediction(num_neighbors=5, distance_method=KNN_Executor.cal_euclidean_distance)
# print("Raw data: \n", model.get_raw_data())
# query_items = model.get_query_items('US251020210019')
# for item in query_items:
#     print(item)
# print("Percentage of accuracy is : ", model.get_accuracy_with_data()*100)

recommend_products = model.make_rating_prediction('US191020210011')
print(f"Recommend {len(recommend_products)} products for user:\n")
for product in recommend_products:
    print(product)