import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Regressor" : KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoosting Classifier" : AdaBoostRegressor()
            }
            params = {
                "decision Tree":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'spliter':['best', 'random'],
                    'max_features': ['sqrt','log2']
                },
                "Random Forest": {
                    'n_estimators': [8,16,32,64,120],
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    'max_features':['sqrt','log2',None]
                },
                "Gradient Boosting" : {
                    'learning_rate': [.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,120]
                },
                "Linear Regression": {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                    'fit_intercept': [True, False],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                },
                "K-Neighbour Regressor":{
                    'n-estimators': [5,7,9,11],
                    'weights':['uniform','distance']
                },
                "XGBRegressor": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                },
                "CatBoosting Regressor":{
                    'iterations': [50,100,150],
                    'learning_rate': [0.01, 0.1],
                    'depth': [4, 6, 8]
                },
                "AdaBoost Regressor":{
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'loss': ['linear', 'square', 'exponential']
                }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                          models = models,param = params)
        
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return best_model_name, r2


        except Exception as e:
            raise CustomException(e,sys)

    