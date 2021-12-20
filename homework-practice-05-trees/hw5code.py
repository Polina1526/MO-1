import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    if len(np.unique(feature_vector)) == 1:
        return None, None, None, None
    # сортировка
    sorted_indexes = np.argsort(feature_vector)
    feature_vector = feature_vector[sorted_indexes]
    target_vector = target_vector[sorted_indexes]
    # расчёт Джини
    R = len(target_vector)
    left_sizes = np.arange(1, R)
    right_sizes = R - left_sizes
    cumsums = np.cumsum(target_vector)
    left_sums = cumsums[:-1]
    right_sums = cumsums[-1] - left_sums
    left_p1 = left_sums / left_sizes
    right_p1 = right_sums / right_sizes
    HR_l = 1 - left_p1**2 - (1 - left_p1)**2
    HR_r = 1 - right_p1**2 - (1 - right_p1)**2
    ginis = (- left_sizes * HR_l - right_sizes * HR_r) / R
    # расчёт порогов
    shape = feature_vector.shape[:-1] + (feature_vector.shape[-1] - 1, 2)
    strides = feature_vector.strides + (feature_vector.strides[-1],)
    thresholds = np.lib.stride_tricks.as_strided(feature_vector, shape=shape, strides=strides) 
    thresholds = np.mean(thresholds, axis=1)
    # выкидываю ненужные разбиения
    idx = np.unique(feature_vector, return_index=True)[1]
    ginis = ginis[idx[1:] - 1]
    thresholds = thresholds[idx[1:] - 1]
    # нахождение лучшего разбиения
    best_id = np.argmax(ginis)
    threshold_best = thresholds[best_id]
    gini_best = ginis[best_id]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):    # поменяла != на ==
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):    # поменяла range(1, sub_X.shape[1]) на range(sub_X.shape[1])
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":    # было Categorical
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count  # поменяла местами current_click и current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                # в первос x[1] поменяла на x[0]
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature]))) # добавила list
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1:    # было len(feature_vector) == 3
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]    # добавила [0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])
        # было sub_y[split]

        
    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        else:
            feature = node["feature_split"]
            if self._feature_types[feature] == "real":
                if x[feature] < node["threshold"]:
                    y_pred = self._predict_node(x, node["left_child"])
                else:
                    y_pred = self._predict_node(x, node["right_child"])
            
            elif self._feature_types[feature] == "categorical":
                if x[feature] in node["categories_split"]:
                    y_pred = self._predict_node(x, node["left_child"])
                else:
                    y_pred = self._predict_node(x, node["right_child"])
            else:
                raise ValueError
                    
        return y_pred
        

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

        
    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    
    
    def get_params(self, deep=False):
        return {'feature_types': self._feature_types}
