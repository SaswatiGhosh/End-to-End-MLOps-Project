import pandas as pd
import os
from sklearn.feature_extraction.text import  TfidfVectorizer
import logging
import yaml

logs_dir= 'logs'
os.makedirs(logs_dir, exist_ok=True)

logger=logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

consolehandler= logging.StreamHandler()
consolehandler.setLevel('DEBUG')

file_handler_path= os.path.join(logs_dir, 'feature_engineering.log')
file_handler=logging.FileHandler(file_handler_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s')
consolehandler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(consolehandler)
logger.addHandler(file_handler)

def load_params(params_path:str):
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError as e:
        logger.error("File not found: %s", params_path)
    except yaml.YAMLError as e:
        logger.error("YAML error:" , e)

    except Exception as e:
        logger.error("Unexpected error %s",e )

def load_data(file_path:str):
    try:
        df= pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the error')
        raise
    except Exception as e:
        logger.error('Unexpected error occur while loading data')
        raise

def apply_tfdif(train_data:pd.DataFrame,test_data:pd.DataFrame, max_features:int) -> tuple:
    try:
        vectorizer= TfidfVectorizer(max_features=max_features) # need more clarity

        x_train= train_data['text'].values
        y_train=train_data['target'].values
        x_test =test_data['text'].values
        y_test=test_data['target'].values

        x_train_bow= vectorizer.fit_transform(x_train)
        x_test_bow=vectorizer.transform(x_test)

        train_df=pd.DataFrame(x_train_bow.toarray())
        train_df['label']=y_train

        test_df=pd.DataFrame(x_test_bow.toarray())
        test_df['label']=y_test

        logger.debug('Bag of words applied and tranformed')
        return train_df, test_df
    
    except Exception as e :
        logger.error('Error during Bag of words applied and tranformed %s',e)
        raise

def save_data(df:pd.DataFrame,file_path:str) ->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e :
        logger.error('Error during Bag of words applied and tranformed')
        raise

def main():
    try:
        params=load_params("params.yaml")
        max_features= params['feature_engineering']['max_features']
        #max_features=50
        train_data=load_data('./data/interim/train_preproccessed.csv')
        test_data=load_data('./data/interim/test_preproccessed.csv')

        train_df, test_df= apply_tfdif(train_data=train_data, test_data=test_data, max_features=max_features)
        save_data(train_df, os.path.join("./data", "processed", "train_tfdif.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfdif.csv"))
    except Exception as e:
        logger.error('Failed to complete the feature eng process %s',e)
        print(f"Error: {e}")

if __name__== '__main__':
    main()


