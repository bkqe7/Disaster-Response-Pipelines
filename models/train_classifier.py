import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt','wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    load all_messages dataframe from database. Define input X and output y.
    input: database name
    output: X, y, category_names
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('all_messages',engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns.tolist()
    return X,y,category_names


def tokenize(text):
    """
    This function nomalizes case and remove punctuation and tokenize and lemmatize text.
    input: text
    output: tokens
    """
    # nomalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]"," ",text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    """
    create pipeline, CountVectorizer, TfidfTransformer and Multioutputclassifier. Use cross validation to optimize 
    the parameters.
    """
    # create pipeline
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer = tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(AdaBoostClassifier()))
    ])
    # set parameters wanted to optimize
    paramaters = {
    'tfidf__use_idf':(True,False),
    'clf__estimator__n_estimators':[50,100],
    'clf__estimator__learning_rate':[0.05,0.1]
    }
    # cross validation to tune parameters
    cv = GridSearchCV(pipeline, param_grid = paramaters,verbose = 2,n_jobs = -1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    this function evaluates the model result.
    input: optimized model, test data, X, Y, and output categories
    output: precision, recall, f1-score and support
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test,Y_pred,target_names = category_names))


def save_model(model, model_filepath):
    """
    export optimized model as a pickle file
    input: model, model file path
    """
    with open(model_filepath,'wb') as file:
        pickle.dump(model,file)
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()