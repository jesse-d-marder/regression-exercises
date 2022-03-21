import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_variable_pairs(df, numerics, categoricals, sample_amt):
    """ Plots pairwise relationships between numeric variables in df along with regression line for each pair. Uses categoricals for hue."""
    # Sampling allows for faster plotting with large datasets at the expense of not seeing all datapoints
    if sample_amt:
        df = df.sample(sample_amt)
    if len(categoricals)==0:
        categoricals = [None]
    for cat in categoricals:    
        for col in numerics:
            for y in numerics:
                if y == col:
                    continue
                sns.lmplot(data = df, 
                           x=col, 
                           y=y, 
                           hue=cat, 
                           palette='Set1',
                           scatter_kws={"alpha":0.2, 's':10}, 
                           line_kws={'lw':4})
            
def months_to_years(df):
    """ Accepts Telco Churn dataframe and converts tenure (months) to tenure_years """
    
    df["tenure_years"] = df.tenure/12
    
    return df

def plot_categorical_and_continuous_vars(df, categorical, continuous, sample_amt):
    """ Accepts dataframe and lists of categorical and continuous variables and outputs plots to visualize the variables"""
    # Sampling allows for faster plotting with large datasets at the expense of not seeing all datapoints
    if sample_amt:
        df = df.sample(sample_amt)
        
    for num in continuous:
        for cat in categorical:
            _, ax = plt.subplots(1,3,figsize=(20,8))
            print(f'Generating plots {num} by {cat}')
            p = sns.swarmplot(data = df, x=cat, y=num, ax=ax[0])
            p.axhline(df[num].mean())
            p = sns.boxplot(data = df, x=cat, y = num, ax=ax[1])
            p.axhline(df[num].mean())
            p = sns.violinplot(data = df, x=cat, y=num, hue = cat, ax=ax[2])
            p.axhline(df[num].mean())
            plt.suptitle(f'{num} by {cat}', fontsize = 18)
            plt.show()