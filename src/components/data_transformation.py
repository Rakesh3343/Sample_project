import sys
import os
from sklearn.compose import ColumnTransformer
from sklearn import pipeline
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logs import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_obj(self):
        try:
            numerical_cols=['reading_score', 'writing_score']
            categorical_cols=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline=Pipeline([('fill_null',SimpleImputer(strategy='mean')),
             ('standardize',StandardScaler())])
            cat_pipeline=Pipeline([('fill_null',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(sparse=False)),('standardize',StandardScaler(),)])
            logging.info('standard scaling is done on numerical columns')
            logging.info("onehot encoding is performed on categorical columns")
            preprocessor=ColumnTransformer([('numerical_pipeline',num_pipeline,numerical_cols),('categorical_pipeline',cat_pipeline,categorical_cols)])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("completed reading train and test data into respective data frames")
            pre_obj=self.get_data_transformer_obj()
            logging.info("fetched preprocessing object")
            # target_col='math_score'
            input_train_target=train_df['math_score']
            input_train_df=train_df.drop(columns=['math_score'],axis=1)
            
            input_test_target=test_df['math_score']
            input_test_df=test_df.drop(columns=['math_score'],axis=1)
            

            logging.info("applying preprocessing object on both train and test")

            input_train_trans=pre_obj.fit_transform(input_train_df)
            input_test_trans=pre_obj.transform(input_test_df)

            train_arr=np.c_[input_train_trans,np.array(input_train_target)]
            test_arr=np.c_[input_test_trans,np.array(input_test_target)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj,obj=pre_obj)

            logging.info("The preprocessor object is saved")

            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj)
        except Exception as e:
            raise CustomException(e,sys)



        
    