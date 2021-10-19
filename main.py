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
