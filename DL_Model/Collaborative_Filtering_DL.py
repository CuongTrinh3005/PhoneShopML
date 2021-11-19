import tensorflow as tf
from tensorflow import keras
import numpy as np

from DL_Model.RecommenderNet import RecommenderNet
from DatabaseConnector import DBConnector


class CF_Model:
    def __init__(self):
        self.df_products, self.df_ratings, self.connector = self.get_dataset_to_train()
        self.user2user_encoded, self.userencoded2user, self.product2product_encoded, self.product_encoded2product = self.normalize_ratings_dataset()
        self.model = self.build_model()

    def get_dataset_to_train(self, driver="SQL Server", servername="QUOC-CUONG", username="sa", password="cuong300599",
                    db_name="OnlinePhoneShop"):
        str_for_connection = "Driver={driver};Server={servername};UID={username};PWD={password};Database={db_name};" \
            .format(driver=driver, servername=servername, username=username, password=password, db_name=db_name)

        connector = DBConnector(servername, username, password, db_name, str_for_connection)
        query_str = "select product_id, product_name, label from dbo.products "
        df_products = connector.query(query_str)

        query_str = "select user_id, product_id, score from dbo.ratings"
        df_ratings = connector.query(query_str)

        # Save to csv file
        df_ratings.to_csv(r'D:\PhoneShopML\data\ratings.csv', header=True, index=True, encoding='utf-8-sig')

        return df_products, df_ratings, connector

    def get_data(self):
        return self.df_products, self.df_ratings

    def normalize_ratings_dataset(self):
        user_ids = self.df_ratings['user_id'].unique().tolist()
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        userencoded2user = {i: x for i, x in enumerate(user_ids)}

        product_ids = self.df_ratings["product_id"].unique().tolist()
        # print("Unique products have been rated: ", len(product_ids))
        product2product_encoded = {x: i for i, x in enumerate(product_ids)}
        product_encoded2product = {i: x for i, x in enumerate(product_ids)}

        self.df_ratings["user"] = self.df_ratings["user_id"].map(user2user_encoded)
        self.df_ratings["product"] = self.df_ratings["product_id"].map(product2product_encoded)

        return user2user_encoded, userencoded2user, product2product_encoded, product_encoded2product

    def prepare_dataset_to_train(self):
        # min and max ratings will be used to normalize the ratings later
        min_rating = min(self.df_ratings['score'])
        max_rating = max(self.df_ratings['score'])

        df_ratings = self.df_ratings.copy()
        df_ratings = df_ratings.sample(frac=1, random_state=42)
        x = df_ratings[["user", "product"]].values
        # Normalize the targets between 0 and 1. Makes it easy to train.
        y = df_ratings["score"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
        # Assuming training on 90% of the data and validating on 10%.
        train_indices = int(0.9 * df_ratings.shape[0])
        x_train, x_val, y_train, y_val = (
            x[:train_indices],
            x[train_indices:],
            y[:train_indices],
            y[train_indices:],
        )
        return x_train, x_val, y_train, y_val

    def build_model(self, embbeding_dims = 50):
        num_users = len(self.user2user_encoded)
        num_products = len(self.product_encoded2product)

        model = RecommenderNet(num_users, num_products, embbeding_dims)
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001)
        )
        x_train, x_val, y_train, y_val = self.prepare_dataset_to_train()
        model.fit(
            x=x_train,
            y=y_train,
            batch_size=64,
            epochs=20,
            verbose=1,
            validation_data=(x_val, y_val),
        )
        model.summary()
        return model

    def make_recommendations(self, user_id, top_products=10):
        if user_id == '' or user_id == None:
            return

        products_rated_by_user = self.df_ratings[self.df_ratings.user_id == user_id]
        products_not_rated = self.df_products[
            ~self.df_products["product_id"].isin(products_rated_by_user.product_id.values)]["product_id"]

        products_not_rated = list(set(products_not_rated).intersection(set(self.product2product_encoded.keys())))
        products_not_rated = [[self.product2product_encoded.get(x)] for x in products_not_rated]

        user_encoder = self.user2user_encoded.get(user_id)
        user_product_array = np.hstack(
            ([[user_encoder]] * len(products_not_rated), products_not_rated)
        )

        ratings = self.model.predict(user_product_array).flatten()
        top_ratings_indices = ratings.argsort()[-top_products:][::-1]
        recommended_product_ids = [
            self.product_encoded2product.get(products_not_rated[x][0]) for x in top_ratings_indices
        ]
        df_recommended_products = self.df_products[self.df_products["product_id"].isin(recommended_product_ids)]
        recommend_products = []
        for row in df_recommended_products.itertuples():
            recommend_products.append([row.product_id, row.product_name, row.label])

        return recommend_products

# model = CF_Model()
# recommended_products = model.make_recommendations(user_id='US281020210063')
# for product in recommended_products: print(product)