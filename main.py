import pandas as pd
import os
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import warnings

my_path = os.getcwd()

def read_pres():
    '''Read in presidential results data'''
    pres_path = os.path.join(my_path, '1976-2016-president.csv')
    presidents = pd.read_csv(pres_path)
    return presidents

presidents = read_pres()

def read_clean_emp():
    '''Read in and clean Bureau of Economic Analysis data'''
    emp_path = os.path.join(my_path, 'state_emp.csv')
    state_emp = pd.read_csv(emp_path, skiprows=4)
    state_emp = state_emp[state_emp['Description'] == 'Per capita net earnings 4/']
    state_emp = state_emp[['GeoName', '2012', '2016']]
    state_emp = pd.melt(state_emp, id_vars=['GeoName'], var_name=['Year'], value_name='percap_net_earn')
    state_emp['Year'] = state_emp['Year'].astype(str).astype(int)
    return state_emp

state_emp = read_clean_emp()

def read_clean_industry():
    industry_path = os.path.join(my_path, 'real_GDP_industry_state.csv')
    industry = pd.read_csv(industry_path, skiprows=4)
    numeric_cols = ['2012:Q1', '2012:Q2', '2012:Q3', '2012:Q4', '2016:Q1', '2016:Q2', '2016:Q3', '2016:Q4']
    industry[numeric_cols] = industry[numeric_cols].apply(pd.to_numeric, errors='coerce', axis=1) # source: https://stackoverflow.com/questions/36814100/pandas-to-numeric-for-multiple-columns
    industry['2012'] = industry['2012:Q1'] + industry['2012:Q2'] + industry['2012:Q3'] + industry['2012:Q4']
    industry['2016'] = industry['2016:Q1'] + industry['2016:Q2'] + industry['2016:Q3'] + industry['2016:Q4']
    industry = industry[['GeoName', 'Description', '2012', '2016']]
    industry = pd.melt(industry, id_vars=['GeoName', 'Description'], var_name='Year', value_name='real_GDP')
    industry = industry.dropna(subset=['Description'])
    industry = pd.pivot(industry, index=['GeoName', 'Year'], columns='Description', values='real_GDP').reset_index()
    industry = industry.rename(columns=lambda x: x.strip()) # source: https://stackoverflow.com/questions/21606987/how-can-i-strip-the-whitespace-from-pandas-dataframe-headers
    industry = industry.drop(columns=['Construction',
                                      'Management of companies and enterprises',
                                      'Transportation and warehousing',
                                      'Utilities',
                                      'Wholesale trade',
                                      'Administrative and support and waste management and remediation services'])
    industry = industry.rename(columns={'Accommodation and food services': 'hospitality',
                                        'Agriculture, forestry, fishing and hunting': 'agriculture',
                                        'Arts, entertainment, and recreation': 'arts',
                                        'Educational services': 'education',
                                        'Finance and insurance': 'finance',
                                        'Health care and social assistance': 'healthcare',
                                        'Mining, quarrying, and oil and gas extraction': 'mining',
                                        'Professional, scientific, and technical services': 'professional services',
                                        'Real estate and rental and leasing': 'real estate',
                                        'Government and government enterprises': 'government',
                                        'Manufacturing': 'manufacturing',
                                        'Retail trade': 'retail trade',
                                        'Information': 'information'})
    industry['Year'] = industry['Year'].astype(str).astype(int)
    return industry

industry = read_clean_industry()

def merged_cleaning():
    '''Merges and cleans dataframe for analysis'''
    pres_merged = presidents.merge(state_emp, left_on=['state', 'year'],
                                              right_on=['GeoName', 'Year'],
                                              how='inner')
    pres_merged = pres_merged.drop(columns=['GeoName', 'Year'])
    pres_merged = pres_merged.merge(industry, left_on=['state', 'year'],
                                              right_on=['GeoName', 'Year'],
                                              how='left')
    pres_merged = pres_merged.drop(columns=['GeoName', 'Year'])
    pres_merged['pct_vote'] = pres_merged['candidatevotes'] / pres_merged['totalvotes']
    return pres_merged

pres_merged = merged_cleaning()

def create_train_test(year):
    '''Create and split training and testing datasets, which will be various
    presidential election years'''
    df = pres_merged[pres_merged['year'] == year]
    maximums = df.groupby(['state'])['pct_vote'].max().reset_index()
    df = df.merge(maximums, left_on='state', right_on='state', how='left')
    df['winner'] = np.where(df['pct_vote_x']==df['pct_vote_y'], 1, 0) # source: https://stackoverflow.com/questions/44067524/creating-a-new-column-depending-on-the-equality-of-two-other-columns
    df = df[df['winner'] == 1]
    df = df[['state', 'party', 'percap_net_earn', 'hospitality',
             'agriculture', 'arts', 'education', 'finance', 'healthcare',
             'information', 'manufacturing', 'mining', 'professional services',
             'real estate', 'retail trade', 'government']]
    df['state'] = df['state'].astype(str)
    df['party'] = df['party'].astype(str)
    x_train_test = df[['percap_net_earn', 'hospitality', 'education', 'finance', 'healthcare',
                       'manufacturing', 'government']]
    y_train_test = df[['party']]
    return df, x_train_test, y_train_test

train_2012, x_train_2012, y_train_2012 = create_train_test(2012)
test_2016, x_test_2016, y_test_2016 = create_train_test(2016)

def test_harness(): # source: lecture
    '''Determine which model will be best to use'''
    models = [('Dec Tree', DecisionTreeClassifier()),
              ('Lin Disc', LinearDiscriminantAnalysis()),
              ('SVC', SVC(gamma='auto')),
              ('Gaussian', GaussianNB())]
    results = []

    for name, model in models:
        kf = StratifiedKFold(n_splits=10)
        res = cross_val_score(model, x_train_2012, y_train_2012, cv=kf, scoring='accuracy')
        res_mean = round(res.mean(), 4)
        res_std  = round(res.std(), 4)
        results.append((name, res_mean, res_std))
    return results

accuracy_results = test_harness()

def fit_model(y_test_df, predict):
    '''Fit model and return results of model'''
    model = DecisionTreeClassifier()
    model.fit(x_train_2012, y_train_2012)
    predict = model.predict(x_test_2016)
    return predict

results_2016 = fit_model(y_test_2016, predict)

def print_class_report(y_test, predict):
    y_class_report = y_test['party']
    warnings.filterwarnings('ignore') # source: https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
    print(classification_report(y_class_report, predict))

report = print_class_report(y_test_2016, predict)

def see_predictions():
    '''Returns dataframes of all states, predictions, and actual results'''
    df_all = y_test_2016
    df_all['state'] = test_2016['state']
    df_all['predictions'] = predict
    df_all = df_all.rename(columns={'party': 'actual'})
    df_all = df_all[['state', 'predictions', 'actual']]
    return df_all

all_predictions_2016 = see_predictions()
