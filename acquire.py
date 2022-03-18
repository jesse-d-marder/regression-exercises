import pandas as pd
from env import get_db_url
import os

def get_telco_data():
    """Return data from telco_churn database in SQL as a pandas data frame"""
    filename = 'telco.csv'
    if os.path.exists(filename):
        print("Using cached data")
        return pd.read_csv(filename)
    
    query = '''
        SELECT *
        FROM customers
        JOIN contract_types
        USING (contract_type_id)
        JOIN internet_service_types
        USING (internet_service_type_id)
        JOIN payment_types
        USING (payment_type_id)'''
    
    # Queries the SQL Database
    df = pd.read_sql(query,get_db_url('telco_churn'))
    df.to_csv(filename, index=False)
    return df

