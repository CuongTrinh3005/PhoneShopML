from DatabaseConnector import DBConnector
import numpy as np
from scipy import optimize


class CFMatrixFactorizer:
    def __init__(self, num_features=3, lambda_var=3, query_user='', n_top=10):
        self.num_features = num_features
        self.lambda_var = lambda_var
        self.query_user = query_user
        self.n_top = n_top
        self.rating_matrix, self.connector = self.get_rating_matrix()
        self.checked_rating_matrix = self.get_checked_rating_matrix()

    def get_rating_matrix(self, driver="SQL Server", servername="QUOC-CUONG", username="sa", password="cuong300599",
                    db_name="OnlinePhoneShop"):
        str_for_connection = "Driver={driver};Server={servername};UID={username};PWD={password};Database={db_name};" \
            .format(driver=driver, servername=servername, username=username, password=password, db_name=db_name)

        connector = DBConnector(servername, username, password, db_name, str_for_connection)
        query_str = "select distinct u.user_id, p.product_id, dbo.fn_get_rating(u.user_id, p.product_id) as score " \
                    "from users u, products p "
        df_users_products_ratings = connector.query(query_str)
        df_users_products_ratings_pivotted = df_users_products_ratings.pivot(index='product_id', columns='user_id',
                                                                             values='score').fillna(0)

        # Save to csv file
        df_users_products_ratings_pivotted.to_csv(r'D:\PhoneShopML\data\rating_matrix.csv', header=True, index=True,
                                                  encoding='utf-8-sig')

        return df_users_products_ratings_pivotted, connector

    def get_checked_rating_matrix(self):
        """This function returns a dataframe with binary values indicating that user has rated for product yet"""
        df_checked_rating = self.rating_matrix.copy()
        column_names = self.rating_matrix.columns.values
        for name in column_names:
            df_checked_rating.loc[df_checked_rating[name] > 0, name] = 1

        return df_checked_rating

    def get_data(self):
        return self.rating_matrix, self.checked_rating_matrix

    def cal_cost_function(self, params, *args):
        Y, R, num_users, num_movies, num_features, lmbda = args
        # Unfold the U and W matrices from params
        X = params[0:num_movies * num_features].reshape(num_movies, num_features, order='F')
        Theta = params[num_movies * num_features:].reshape(num_users, num_features, order='F')

        squared_error = np.power(np.dot(X, Theta.T) - Y, 2)
        # for cost function, sum only i,j for which R(i,j)=1
        J = (1 / 2.) * np.sum(squared_error * R)
        J = J + (lmbda / 2.) * (np.sum(np.power(Theta, 2)) + np.sum(np.power(X, 2)))

        return J

    def cal_gradient_decent(self, params, *args):
        Y, R, num_users, num_movies, num_features, lmbda = args
        # Unfold the U and W matrices from params
        X = params[0:num_movies * num_features].reshape(num_movies, num_features,  order='F')
        Theta = params[num_movies * num_features:].reshape(num_users, num_features,  order='F')

        X_grad_reg = lmbda * X
        Theta_grad_reg = lmbda * Theta

        X_grad = np.dot((np.dot(X, Theta.T) - Y) * R, Theta) + X_grad_reg
        Theta_grad = np.dot(((np.dot(X, Theta.T) - Y) * R).T, X) + Theta_grad_reg

        grad = np.hstack([X_grad.flatten('F'), Theta_grad.flatten('F')])

        return grad

    def normalize_rating(self, Y, R):
        m, n = np.shape(Y)
        Ymean = np.zeros((m, 1))
        Ynorm = np.zeros((m, n))
        for i in range(m):
            idx = np.nonzero(R[i, :] == 1.0)[0]
            Ymean[i] = Y[i, idx].mean(axis=0)
            Ynorm[i, idx] = Y[i, idx] - Ymean[i]
        return Ymean, Ynorm

    def build_model(self):
        Y = self.rating_matrix.to_numpy()
        R = self.checked_rating_matrix.to_numpy()

        Ymean, Ynorm = self.normalize_rating(Y, R)
        num_products, num_users = np.shape(Y)

        X = np.random.standard_normal(num_products * self.num_features)
        Theta = np.random.standard_normal(num_users * self.num_features)

        initial_parameters = np.hstack([X, Theta])
        args = (Ynorm, R, num_users, num_products, self.num_features, self.lambda_var)

        theta = optimize.fmin_cg(self.cal_cost_function, initial_parameters, fprime=self.cal_gradient_decent, args=args, maxiter=100)

        # Unfold the returned theta back into X and Theta
        X = theta[0:num_products * self.num_features].reshape(num_products, self.num_features, order='F')
        Theta = theta[num_products * self.num_features:].reshape(num_users, self.num_features, order='F')

        return X, Theta

    def make_prediction_matrix(self):
        X, Theta = self.build_model()
        return np.dot(X, Theta.T)

    def make_prediction_for_user(self, exclude_rated=False):
        user_id = self.query_user
        user_index_in_matrix = self.rating_matrix.columns.get_loc(user_id)
        print("Index of user in matrix: ", user_index_in_matrix)

        Ymean, _ = self.normalize_rating(self.rating_matrix.to_numpy(), self.checked_rating_matrix.to_numpy())
        prediction_matrix = self.make_prediction_matrix()

        user_prediction = prediction_matrix[:, user_index_in_matrix] + Ymean.flatten()
        user_prediction = [x for x in user_prediction if str(x) != 'nan']

        # Sort user prediction with descending order of score
        sorted_indices = np.argsort(user_prediction)[::-1]
        list_products = self.rating_matrix.index.values.tolist()
        print("Collaborative Filtering recommendation products:\n")

        if exclude_rated is True:
            rated_products = self.get_rated_products_of_user(user_id)
        recommend_products = []
        for position in range(self.n_top):
            index = sorted_indices[position]
            recommend_product_id = list_products[index]
            recommend_product_name = self.get_product_name(recommend_product_id)
            predicted_score = user_prediction[index]
            info = {'product_id': recommend_product_id, 'product_name: ': recommend_product_name, 'predict_score: ': predicted_score}
            # Re-ranking
            if exclude_rated is True:
                if recommend_product_id not in rated_products:
                    recommend_products.append(info)

            else: recommend_products.append(info)
        return recommend_products

    def get_product_name(self, product_id):
        query_str = f"select product_name from products p where product_id='{product_id}'"
        df_name = self.connector.query(query_str)
        return df_name.values.tolist()[0][0]

    def get_rated_products_of_user(self, user_id):
        query_str = f"select product_id from ratings p where user_id='{user_id}'"
        df_results = self.connector.query(query_str)
        list_ids = []
        for item in df_results.values.tolist():
            list_ids.append(item[0])
        return list_ids

# cf_model = CFMatrixFactorizer(query_user='US281020210061')
# recommend_products = cf_model.make_prediction_for_user()
# for produtct in recommend_products:
#     print(produtct)