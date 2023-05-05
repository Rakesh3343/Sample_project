import os
import pickle
from src.exception import CustomException
import sys
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV



def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as f:
            pickle.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,X_test,y_train,y_test,models,params):
    try:
        report={}
        for name,model in models.items():
            gs=GridSearchCV(estimator=model,param_grid=params[name],scoring='r2',cv=3,verbose=1)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            train_model_score=r2_score(y_train_pred,y_train)
            test_model_score=r2_score(y_test,y_test_pred)
            report[name]=test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)

