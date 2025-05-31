import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score, recall_score,roc_auc_score
import logging

logs_dir='logs'
os.makedirs(logs_dir,exist_ok=True)

logger=logging.getLogger()
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

logs_file_path=os.path.join(logs_dir,'model_evaluation.log')
file_handler=logging.FileHandler(logs_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter( '%(asctime)s - %(name)s -%(levelname)s - %(message)s' )
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path:str):
    try:
        with open (file_path, 'rb') as file:
            model=pickle.load(file)
        logger.debug('Model has been loaded from %s ', file_path)
        return model
    except FileNotFoundError as e:
        logger.error('File not found error %s', e)
        raise
    except Exception :
        logger.error('Unexpected error has occured', e)
        raise

def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logger.debug('Data is loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the error %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error has occured', e)
        raise
def evaluate_model(clf, x_test:np.ndarray,y_test :np.ndarray)->dict:
    try:
        y_pred= clf.predict(x_test)
        y_pred_proba=clf.predict_proba(x_test)[:,1]

        accuracy=accuracy_score(y_test,y_pred)
        precision= precision_score(y_test,y_pred)
        recall= recall_score(y_test,y_pred)
        auc= roc_auc_score(y_test, y_pred_proba)

        metrics_dict={
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
            }
        logger.debug('Model evaluation calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation')
        raise

def save_metrics(metrics: dict, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path ), exist_ok=True)
        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4) #doubt
        logger.debug('Metrics saved to %s', file_path)
    except  Exception as e:
        logger.debug('Unexplcted error has occured')
        raise

def main():
    try:
        clf=load_model('./models/models.pkl')
        test_data=load_data('./data/processed/test_tfdif.csv')

        x_test=test_data.iloc[:,:-1].values
        y_test=test_data.iloc[:,-1].values

        metrics=evaluate_model(clf,x_test,y_test)
        save_metrics(metrics,'./reports/reports.json')
    except Exception as e:
        logger.error('Unable to evaluate %s',e)
        raise

if __name__=='__main__':
    main()




        


