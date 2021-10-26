import math
from statistics import mode


class KNN_Executor:
    def __init__(self, data, query, k, distance_fn, choice_fn, match_exactly=False):
        self.data = data
        self.query = query
        self.k = k
        self.distance_fn = distance_fn
        self.choice_fn = choice_fn
        self.match_exactly = match_exactly

    @classmethod
    def mean(cls, labels):
        return sum(labels) / len(labels)

    @classmethod
    def mode(cls, labels):
        """This function returns the item having the most appearance"""
        return mode(labels)
        # return Counter(labels).most_common(1)[0][0]

    @classmethod
    def cal_manhattan_distance(cls, point1, point2):
        sum_distance = 0
        for index in range(len(point1)):
            sum_distance +=abs(point1[index] - point2[index])

        return sum_distance

    @classmethod
    def cal_euclidean_distance(cls, point1, point2):
        sum_squared_distance = 0
        # print("point 1: ", point1, " and its shape: ", len(point1))
        # print("point 2: ", point2, " and its shape: ", len(point2))
        for index in range(len(point1)):
            sum_squared_distance += math.pow(point1[index] - point2[index], 2)

        return math.sqrt(sum_squared_distance)

    @classmethod
    def cal_hassanat_distance(cls, point1, point2):
        distance = 0
        for xi, yi in zip(point1, point2):
            min_xi_yi = min(xi, yi)
            max_xi_yi = max(xi, yi)
            if min_xi_yi >= 0:
                numerator = 1 + min_xi_yi
                dominator = 1 + max_xi_yi
            else:
                numerator = 1 + min_xi_yi + abs(min_xi_yi)
                dominator = 1 + max_xi_yi + abs(min_xi_yi)
            distance += 1 - numerator / dominator

        return distance

    @property
    def inference(self):
        neighbor_distances_and_indices = []

        # 3. For each example in the data
        for index, example in enumerate(self.data):
            # 3.1 Calculate the distance between the query example and the current
            # example from the data.
            distance = self.distance_fn(example[:-1], self.query)

            # 3.2 Add the distance and the index of the example to an ordered collection
            if self.match_exactly is True:
                neighbor_distances_and_indices.append((distance, index))
            else:
                if distance != 0:
                    neighbor_distances_and_indices.append((distance, index))

        # 4. Sort the ordered collection of distances and indices from
        # smallest to largest (in ascending order) by the distances
        sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)

        # 5. Pick the first K entries from the sorted collection
        k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:self.k]

        # 6. Get the labels of the selected K entries
        k_nearest_labels = [self.data[i][-1] for distance, i in k_nearest_distances_and_indices]

        # 7. If regression (choice_fn = mean), return the average of the K labels
        # 8. If classification (choice_fn = mode), return the mode of the K labels
        return k_nearest_distances_and_indices, self.choice_fn(k_nearest_labels)

# Test class methods
# labels = [1, 2, 1, 0, 0, 2, 1, 2]
# print("Mean methods: ", KNN_Executor.mean(labels))
# print("\nMode methods: ", KNN_Executor.mode(labels))
#
# # Regression problem
# reg_data = [
#     [65.75, 112.99],
#     [71.52, 136.49],
#     [69.40, 153.03],
#     [68.22, 142.34],
#     [67.79, 144.30],
#     [68.70, 123.30],
#     [69.80, 141.49],
#     [70.01, 136.46],
#     [67.90, 112.37],
#     [66.49, 127.45],
# ]
#
# # Question:
# # Given the data we have, what's the best-guess at someone's weight if they are 60 inches tall?
# reg_query = [60]
# data = reg_data
# k=5
# distance_method = KNN_Executor.cal_euclidean_distance
# choice = KNN_Executor.mean
#
# knn_model = KNN_Executor(data=data, query=reg_query, k=k, distance_fn=distance_method, choice_fn=choice)
# reg_k_nearest_neighbors, reg_prediction = knn_model.inference()
# print("Nearest neighbors: \n", reg_k_nearest_neighbors)
# print("Prediction labels: \n", reg_prediction)
#
# # Classification problem
# print("\nClassification problem:\n")
# clf_data = [
#     [22, 1],
#     [23, 1],
#     [21, 1],
#     [18, 1],
#     [19, 1],
#     [25, 0],
#     [27, 0],
#     [29, 0],
#     [31, 0],
#     [45, 0],
# ]
# clf_query = [33]
# choice_clf = KNN_Executor.mode
#
# knn_model = KNN_Executor(data=clf_data, query=clf_query, k=3, distance_fn=distance_method, choice_fn=choice_clf)
# clf_k_nearest_neighbors, clf_prediction = knn_model.inference()
# print("Nearest neighbors: ",  clf_k_nearest_neighbors , " and predictions: ", clf_prediction)
