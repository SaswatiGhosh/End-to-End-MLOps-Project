import numpy as np
import pandas as pd
import os
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

logs_dir ='logs'
os.makedirs(logs_dir, exist_ok=True)

logger=logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path= os.path.join(logs_dir, 'model_training.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError as e:
        logger.error("File not found: %s", params_path)
    except yaml.YAMLError as e:
        logger.error("YAML error:" , e)

    except Exception as e:
        logger.error("Unexpected error %s",e )

def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the file %s', e)
        raise
    except FileNotFoundError as e:
        logger.error(' File not found %s', e)
        raise
    except Exception:
        logger.error('Unexpected error occured %s', e)
        raise
    

def train_model(x_train:np.ndarray,y_train:np.ndarray,params) -> RandomForestClassifier: #doubt
    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError('The number of samples in x_train and y_train to be same')
        logger.debug('Initializing RandomForestClassifier has started')
        clf=RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        logger.debug('Model training started with %d samples', x_train.shape[0])
        clf.fit(x_train,y_train)                                 # It learns patterns from the data.
        logger.debug('Model Training completed')
        return clf
    except ValueError as e:
        logger.error('ValueError during model training')
        raise
    except Exception as e:
        logger.error('Unexpected error occured %s',e)
        raise


def save_model(model, file_path:str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)                                              #doubt
        logger.debug('Model save to %s ', file_path)
    except FileNotFoundError as e:
        logger.error(' File not found %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured %s',e)
        raise

def main():
    try:
        params=load_params('params.yaml')['model_training']
    
        #params={'n_estimators':25, 'random_state': 2}
        train_data=load_data('./data/processed/train_tfdif.csv')
        x_train= train_data.iloc[:, :-1].values
        y_train=train_data.iloc[: , -1].values

        clf= train_model(x_train=x_train, y_train=y_train, params=params)
        model_save_path='models/models.pkl'
        save_model(clf, model_save_path)
    except Exception as e :
        logger.error('Failed to complete the model building process %s', e)
        print('unable to complete the model training')
        raise

if __name__=='__main__':
    main()



