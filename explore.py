import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_variable_pairs(df, numerics):
    """ Plots pairwise relationships between numeric variables in df along with regression line for each pair"""
    
    for col in numerics:
        for y in numerics:
            if y == col:
                continue
            sns.lmplot(data = df[numerics], x=col, y=y, line_kws={'color': 'red'})
            
def months_to_years(df):
    """ Accepts Telco Churn dataframe and converts tenure (months) to tenure_years """
    
    df["tenure_years"] = df.tenure/12
    
    return df

def plot_categorical_and_continuous_vars(df, categorical, continuous):
    """ Accepts dataframe and lists of categorical and continuous variables and outputs plots to visualize the variables"""
    
    for num in continuous[0:3]:
        _, ax = plt.subplots(1,3,figsize=(20,8))
        for cat in categorical[0:1]:
            print(f'Generating plots {num} by {cat}')
            sns.swarmplot(data = df, x=cat, y=num, ax=ax[0])
            sns.boxplot(data = df, x=cat, y = num, ax=ax[1])
            sns.violinplot(data = df, x=cat, y=num, hue = cat, ax=ax[2])
            plt.suptitle(f'{num} by {cat}', fontsize = 18)
        plt.show()