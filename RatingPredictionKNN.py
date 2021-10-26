import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

        exclude_cols = ['product_id', 'product_name', 'gender', 'unit_price', 'discount']
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
        query_str = "select p.product_id, p.product_name, p.ram, p.rom, p.battery_power, p.resolution, p.max_core," \
                    "p.max_speed, p.refresh_rate, p.sim_support, p.networks, p.no_front_cam, p.touch_screen, p.wifi, p.bluetooth," \
                    "p.label, p.compatible_devices, p.functions, p.warranty " \
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
        query_items = self.get_query_items(userId)
        recommend_products = []
        for index, item in enumerate(query_items):
            knn_model = KNN_Executor(data=data, query=item, k=self.num_neighbors,
                                     distance_fn=self.distance_method
                                     , choice_fn=KNN_Executor.mean, match_exactly=True)
            k_nearest_neighbors, rating_prediction = knn_model.inference
            # Round off rating to nearest 0.5. Ex: 2.3 -> 2.5
            rating_round_off = round(rating_prediction * 2) / 2
            if rating_round_off >= rating_criteria_score:
                print("Rating predictions: ", rating_round_off)
                item_to_recommend = self.list_id_name[index]
                print(f"Product info: {item_to_recommend}")
                recommend_products.append(item_to_recommend)

        return recommend_products

    def get_accuracy_with_manual_knn(self, dataset):
        data_train = []; labels = []
        for item in dataset:
            data_point = item[:-1]
            label = item[-1]
            data_train.append(data_point)
            labels.append(label)

        st_x = StandardScaler()
        scale_data_train = st_x.fit_transform(data_train)
        scale_data_train = scale_data_train.tolist()
        for index in range(len(scale_data_train)):
            scale_data_train[index].append(labels[index])

        x_train, x_test, y_train, y_test = train_test_split(scale_data_train, labels, test_size=0.1, random_state=0)

        predictions = []
        for index, item in enumerate(x_test):
            knn_model = KNN_Executor(data=x_train, query=item, k=self.num_neighbors,
                                     distance_fn=self.distance_method
                                     , choice_fn=KNN_Executor.mode, match_exactly=True)
            k_nearest_neighbors, rating_prediction = knn_model.inference
            predictions.append(rating_prediction)

        accuracy = accuracy_score(y_test, predictions) * 100
        print(f"Accuracy with manual scaler knn: {accuracy}%")

    def make_rating_prediction_with_scaler(self, userId, rating_criteria_score=3.5):
        # Preprocessed data
        self.data = self.preprocess_data(self.data)

        # Extract processed data
        df_without_name_and_id = self.data[self.data.columns[2:]]

        # Extracting Independent and dependent Variable
        x = df_without_name_and_id.iloc[:, :-1].values
        y = df_without_name_and_id.iloc[:, -1].values

        # Use label encoder
        lab_enc = preprocessing.LabelEncoder()
        encoded_label = lab_enc.fit_transform(y)

        st_x = StandardScaler()
        x_train = st_x.fit_transform(x)

        x_train = x_train.tolist()
        for index in range(len(x_train)):
            x_train[index].append(encoded_label[index])

        self.get_accuracy_with_manual_knn(x_train)

        query_items = self.get_query_items(userId)
        scaled_items = st_x.transform(query_items)
        scaled_items = scaled_items.tolist()

        recommend_products = []
        for index, item in enumerate(scaled_items):
            knn_model = KNN_Executor(data=x_train, query=item, k=self.num_neighbors,
                                     distance_fn=self.distance_method
                                     , choice_fn=KNN_Executor.mode, match_exactly=True)
            k_nearest_neighbors, rating_prediction = knn_model.inference
            prediction=lab_enc.inverse_transform([rating_prediction])

            rating_round_off = round(prediction[0] * 2) / 2
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
        lab_enc = preprocessing.LabelEncoder()
        encoded_label = lab_enc.fit_transform(y)

        # Splitting the dataset into training and test set.
        x_train, x_test, y_train, y_test = train_test_split(x, encoded_label, test_size=0.1, random_state=0)

        # feature Scaling
        st_x = StandardScaler()
        x_train = st_x.fit_transform(x_train)
        x_test = st_x.transform(x_test)

        # Fitting K-NN classifier to the training set
        from sklearn.neighbors import KNeighborsClassifier
        # classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=1)
        classifier = KNeighborsClassifier(n_neighbors=5, metric=lambda a, b: KNN_Executor.cal_hassanat_distance(a, b))
        classifier.fit(x_train, y_train)

        # Predicting the test set result
        y_pred = classifier.predict(x_test)
        print("y prediction: ", lab_enc.inverse_transform(y_pred))

        # Creating the Confusion matrix
        from sklearn.metrics import confusion_matrix
        # cm = confusion_matrix(y_test, y_pred)
        # print("Confusion matrix:\n", cm)

        # Get the accuracy
        return accuracy_score(y_test, y_pred)


model = KNN_Rating_Prediction(num_neighbors=5, distance_method=KNN_Executor.cal_hassanat_distance)
recommend_products = model.make_rating_prediction_with_scaler('US251020210018')
print(f"Recommend {len(recommend_products)} products for user:\n")
for product in recommend_products:
    print(product)