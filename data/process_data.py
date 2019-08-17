import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    """
    This function load message and category data and combine them
    input: messages filepath and categories filepath
    output: a dataframe combines messages and categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # join two datasets on id
    df = pd.merge(messages,categories,on='id')
    return df


def clean_data(df):
    """
    This function splits the categories into seperate category columns with value 0 and 1.
    Clean column values and remove duplicates.
    input: combined dataframe
    output: cleaned dataframe with seperated category columns    
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand = True) 
    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x:x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # noticed 'related' column has 3 values, change 2 to 0
    categories['related'].replace(to_replace=2,value = 0,inplace = True)
    # child_alone column only has one value. delete this column
    categories.drop(labels='child_alone',axis = 1, inplace = True)
    # drop the original categories column from `df`
    df.drop(columns =['categories'],axis =1, inplace =True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis =1)
    #  remove duplicates
    df.drop_duplicates(inplace = True)
    return df

def save_data(df, database_filename):
    """
    this function saves the dataset to a database.
    input: dataset and database name
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('all_messages', engine, index=False)
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()