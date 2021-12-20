from collections import defaultdict

import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin


sns.set(style='darkgrid')


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting(BaseEstimator, ClassifierMixin):

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.loss_derivative2 = lambda y, z: y ** 2 * self.sigmoid(-y * z) * (1 - self.sigmoid(-y * z))

        
    def fit_new_base_model(self, x, y, predictions):
        bootstrap_size = int(self.subsample * x.shape[0])
        bootstrap_ind = np.random.randint(low=0, high=x.shape[0], size=bootstrap_size)
        
        residuals = - self.loss_derivative(y[bootstrap_ind], predictions[bootstrap_ind])
        
        model = self.base_model_class(**self.base_model_params)
        model.fit(x[bootstrap_ind], residuals)
        
        new_predictions = model.predict(x[bootstrap_ind])
        best_gamma = self.find_optimal_gamma(y[bootstrap_ind], predictions[bootstrap_ind], new_predictions)
        
        self.gammas.append(self.learning_rate * best_gamma)
        self.models.append(model)

        
    def fit(self, x_train, y_train, x_valid=None, y_valid=None):  # =None я добавила, когда пыталась отдебажить блэндинг
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        if x_valid is not None:  # этот и все подобные ифы ниже я добавила, когда пыталась отдебажить блэндинг
            valid_predictions = np.zeros(y_valid.shape[0])
        
        self.history['train'] = [self.loss_fn(y_train, train_predictions)]
        if x_valid is not None:
            self.history['valid'] = [self.loss_fn(y_valid, valid_predictions)]

        ind = 0
        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.gammas[-1] * self.models[-1].predict(x_train)
            if x_valid is not None:
                valid_predictions += self.gammas[-1] * self.models[-1].predict(x_valid)
            
            self.history['train'].append(self.loss_fn(y_train, train_predictions))
            if x_valid is not None:
                self.history['valid'].append(self.loss_fn(y_valid, valid_predictions))
            
            if self.early_stopping_rounds is not None:
                self.validation_loss[ind] = self.history['valid'][-1]
                if np.all(self.validation_loss == self.validation_loss[0]):
                    break
                ind = (ind + 1) % self.early_stopping_rounds    

        if self.plot:
            _x = np.arange(self.n_estimators + 1)
            plt.figure(figsize=(10,8))
            
            sns.lineplot(x=_x, y=self.history['train'], label='train')
            sns.lineplot(x=_x, y=self.history['valid'], label='valid')
            
            plt.title('Losses')
            plt.xlabel('n estimators')
            plt.ylabel('loss')
            plt.legend()
            plt.show()

            
    def predict_proba(self, x):
        predictions = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            predictions += gamma * model.predict(x)
        
        predictions = np.array(self.sigmoid(predictions))
        return np.vstack([1 - predictions, predictions]).T  # здесь из-за транспонирования нужно изначально делать в обратно порядке

            
    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    
    def score(self, x, y):
        return score(self, x, y)

    
    @property
    def feature_importances_(self):      
        feature_importances = self.models[0].feature_importances_
        for i in range(1, self.n_estimators):
            feature_importances += self.models[i].feature_importances_
        
        feature_importances /= self.n_estimators
        return feature_importances
