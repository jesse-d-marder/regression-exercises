import os
import pandas as pd
from sklearn.model_selection import train_test_split

def wrangle_zillow():
    
    # Acquire data from CSV if exists
    if os.path.exists('zillow_2017.csv'):
        print("Using cached data")
        df = pd.read_csv('zillow_2017.csv')
    # Acquire data from database if CSV does not exist
    else:
        print("Acquiring data from server")
        query = """
            SELECT parcelid,bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
            FROM properties_2017
            JOIN propertylandusetype
            USING (propertylandusetypeid)
            WHERE propertylandusedesc = "Single Family Residential";
            """
        df = pd.read_sql(query, get_db_url('zillow'))
        df.to_csv('zillow_2017.csv', index=False)
    
    # Prepare the data for exploration and modeling
    # Rename columns as needed
    df=df.rename(columns = {'parcelid':'parcel_id', 
                            'bedroomcnt':'bedroom_cnt', 
                            'bathroomcnt':'bathroom_cnt', 
                            'calculatedfinishedsquarefeet':'square_feet',
                            'taxvaluedollarcnt':'tax_value', 
                            'yearbuilt':'year_built', 
                            'taxamount':'tax_amount'})
    
    # Drops the rows with Null values, representing a very small percentage of the dataset (<0.6%)
    df = df.dropna()
    
    # Convert year built column to integer from float
    df.year_built = df.year_built.astype('int64')
    df.bedroom_cnt = df.bedroom_cnt.astype('int64')


    return df

def split_data(df, train_size_vs_train_test = 0.8, train_size_vs_train_val = 0.7, random_state = 123):
    """Splits the inputted dataframe into 3 datasets for train, validate and test (in that order).
    Can specific as arguments the percentage of the train/val set vs test (default 0.8) and the percentage of the train size vs train/val (default 0.7). Default values results in following:
    Train: 0.56
    Validate: 0.24
    Test: 0.2"""
    train_val, test = train_test_split(df, train_size=train_size_vs_train_test, random_state=123)
    train, validate = train_test_split(train_val, train_size=train_size_vs_train_val, random_state=123)
    
    train_size = train_size_vs_train_test*train_size_vs_train_val
    test_size = 1 - train_size_vs_train_test
    validate_size = 1-test_size-train_size
    
    print(f"Data split as follows: Train {train_size:.2%}, Validate {validate_size:.2%}, Test {test_size:.2%}")
    
    return train, validate, test