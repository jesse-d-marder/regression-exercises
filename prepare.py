import pandas as pd
import numpy as np
from env import get_db_url
from sklearn.model_selection import train_test_split

def prep_telco(df):
    """ Reads in a dataframe with Telco Churn data and returns a prepared df for further exploration and modeling"""
    # Drop unnecessary foreign key ids
    df = df.drop(columns=['payment_type_id','internet_service_type_id','contract_type_id'])
    # From prior exploration of dataset a small number of the total_charges are just whitespace - these are all new customers who haven't been with the company for >1 month.
    # Given that it is a very small proportion of the total dataset these rows will be deleted for ease of computation later on
    df = df.drop(df[df.total_charges == " "].index)
    # Convert total_charges to float for later analysis
    df.total_charges = df.total_charges.astype('float64')
    
    # Encode categorical variables
    df = df.rename(columns={'senior_citizen':'is_senior_citizen'})
    df.churn = df.churn.map({'Yes':1,'No':0})
    df['is_male'] = df.gender.map({'Male':1,'Female':0})
    df['has_phone'] = df.phone_service.map({'Yes':1,'No':0})
    df['has_internet_service'] = df.internet_service_type.map({'Fiber optic':1,'DSL':1,'None':0})
    df['has_partner']= df.partner.map({'Yes':1,'No':0})
    df['has_dependent'] = df.dependents.map({'Yes':1,'No':0})
    df['is_paperless'] = df.paperless_billing.map({'Yes':1,'No':0})
    df['is_month_to_month'] = df.contract_type.map({'Two year':0,'One year':0, 'Month-to-month':1})
    df['is_autopay'] = df.payment_type.map({'Electronic check': 0, 'Mailed check': 0, 'Bank transfer (automatic)':1, 'Credit card (automatic)': 1})
    df['has_streaming'] = ((df.streaming_movies == 'Yes')|(df.streaming_tv == 'Yes'))
    
    # Remove unnecessary columns after encoding
    df = df.drop(columns=['gender','phone_service','dependents','partner','paperless_billing'])
    
    return df

def train_validate_test_split(df, target, seed=123):
    '''
    Cleanly splits data into train, validate, and test sets.
    Takes as arguments:
    dataframe, the name of the target variable, and an integer for setting a random seed.
    Test is set to 20% of the original dataset, validate is 24% of the 
    original dataset, and train is set to 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

