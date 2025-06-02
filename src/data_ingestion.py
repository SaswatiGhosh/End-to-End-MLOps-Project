import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

log_dirs="logs"
os.makedirs(log_dirs, exist_ok=True) #ensure logs directory exists, if doesnt exist, then creates

#logging configuration
logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler= logging.StreamHandler()
logger.setLevel('DEBUG')

log_file_path= os.path.join('logs', 'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
logger.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url:str) -> pd.DataFrame:
    """ Load data from csv file"""
    try:
        df=pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.debug("Failed to parse the csv file %s", e )
        raise
    except Exception as e:
        logger.debug("Unexpected error occured as an Exception %s", e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data"""
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
        df.rename(columns={'v1': 'target', 'v2':'text'}, inplace=True)
        logger.debug("Data preprocessing completed")
        return df
        
    except KeyError as e:
        logger.error("Missing Column %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected excetion has occured %s " , e)
        raise

def save_data(train_data:pd.DataFrame , test_data:pd.DataFrame, data_path:str):
    try:
        raw_data_path=os.path.join(data_path, 'raw').replace('\\','/')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'), index=False)
        
        logger.debug("Train and data saved in %s", raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occured %s ', e)
        raise

def main():
    try:
        test_size=0.2
        data_path="https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv"
        df=load_data(data_url=data_path)
        final_df=preprocess_data(df)
        # print(final_df)
        train_data, test_data= train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data=train_data, test_data=test_data, data_path='./data')
        print("Execution completed")
    except Exception as e:
        logger.error('Failed to complete data ingestion %s ',e )
        raise #why do we use

if __name__ == '__main__':
    main()


