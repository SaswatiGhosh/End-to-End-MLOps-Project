import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import  string
import nltk
nltk.download('stopwords')
# nltk.download('punkt')

logs_dir= 'logs'
os.makedirs(logs_dir, exist_ok=True)

logger=logging.getLogger('preprocessing')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler_path=os.path.join(logs_dir, 'pre_processing.log')
file_handler=logging.FileHandler(file_handler_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(acstime)s - %(name)s -%(levelname)s -%(message)s')

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    ps= PorterStemmer() #purpose
    text=text.lower()
    text=nltk.word_tokenize(text)
    text =[word for word in text if word.isalnum()]
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text=[ps.stem(word) for word in text]
    return " ".join(text)  #why

def preprocess_df(df, text_column='text', target_column='target'):
    try:
        logger.debug("Starting Preprocessing")
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column]) #cant understand-LabelEncoder` from `sklearn` converts **categorical labels into numeric values*
        logger.debug("Target column encoded")

        df.loc[:, text_column]= df[text_column].apply(transform_text) #Update the entire column with the cleaned version -details in .txt file
        logger.debug("Text column transform")
        return df
    except KeyError as e:
        logger.error('Column not found %s', e)
        raise
    except Exception as e:
        logger.error('Error during normalization %s', e)
        raise
def main(text_column='text', target_column='target'):
    try:
        # print('Finally into main function')
        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv('./data/raw/test.csv')
        logger.debug(" Data loaded properly")

        train_preproccessed_data=preprocess_df(train_data, text_column,target_column)
        test_preproccessed_data=preprocess_df(test_data, text_column,target_column)

        data_path=os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok=True)
        train_preproccessed_data.to_csv(os.path.join(data_path,"train_preproccessed.csv"), index=False)
        test_preproccessed_data.to_csv(os.path.join(data_path,"test_preproccessed.csv"), index=False)
    except FileNotFoundError as e:
        logger.error("File not found %s", e)
    except pd.errors.EmptyDataError as e:
        logger.error("No data %s", e)
    except Exception as e:
        logger.error("Failed to complete preprocessing %s", e)
        print(f"Error : {e}")


if __name__== '__main__':
    main()

#train_preprocecssed data and test_processed data is created
