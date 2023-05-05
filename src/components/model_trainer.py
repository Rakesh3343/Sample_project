from sklearn.linear_model import LinearRegression
import numpy as np
import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logs import logging
from src.utils import evaluate_models,save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_training(self,train_array,test_array,preprocessor_path):
        try:
            X_train,y_train,X_test,y_test=train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
            models={'linear regression':LinearRegression(),'decisiontree regression':DecisionTreeRegressor(),'XG boost':XGBRegressor(),
                    'catboost':CatBoostRegressor(),'adaboost':AdaBoostRegressor(),'gradientboost':GradientBoostingRegressor(),'randomforest':RandomForestRegressor()}
            report=evaluate_models(X_train,X_test,y_train,y_test,models)
            models_keys=list(models.keys())
            best_score=np.max(list(report.values()))
            best_model_name=models_keys[np.argmax(list(report.values()))]
            best_model=models[best_model_name]
            if best_score<0.60:
                raise CustomException("No model is performing good")
            logging.info("Best model found!")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            return best_score,best_model_name
        except Exception as e:
            raise CustomException(e,sys)
        
            
            