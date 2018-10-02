import numpy as np
import pandas as pd
import os 
import tensorflow as tf
import matplotlib.pyplot as plt

num_train = 813160
num_test = 892816


def safe_mkdir(path):
#Create a directory if there isn't one already
    try:
        os.mkdir(path)
    except OSError:
        pass

def read_data(train_path,test_path):
    #read data from given path
    raw_train_data = pd.read_csv(train_path)
    raw_test_data = pd.read_csv(test_path)   
    return raw_train_data , raw_test_data


def oversampling(raw_train_data):
    #duplicate class 1 samples 10 times
    index = list(raw_train_data[raw_train_data['target']==1].index)
    class1_data = raw_train_data.iloc[index]
    lst = [class1_data] * 10
    class1_data = pd.concat(lst)
    lst = [raw_train_data,class1_data]
    raw_train_data = pd.concat(lst)
    return raw_train_data

def combine_train_test(raw_train_data,raw_test_data):
    data = [raw_train_data , raw_test_data]
    data = pd.concat(data)
    return data

def prepare_data(train_path,test_path):
    raw_train_data , raw_test_data = read_data(train_path,test_path)
    raw_train_data = oversampling(raw_train_data)
    target = raw_train_data['target'].values
    raw_train_data = raw_train_data.drop(columns = ['target'])
    test_id = raw_test_data['id'].values
    data = combine_train_test(raw_train_data,raw_test_data)
    return data, target ,test_id
    pass

def one_hot_encoding(data,column_names):
    data = pd.get_dummies(data , columns = column_names , prefix = column_names)
    return data  


def normalize_values(data,column_names): 
    num_min = data[column_names].min()  
    num_max = data[column_names].max()
    norm_val = num_max - num_min
    data[column_names] = (data[column_names] - num_min) / norm_val
    return data


def preprocess_data(data):
    column_names = list(data.columns.values)
    num_columns = column_names[1:24]
    #der_columns = column_names[24:43]
    der_numerical_columns = column_names[24:27]
    der_categorical_columns = column_names[27:43]
    cat_columns = column_names[43:]

    #handling missing values in numerical variables 
    data[num_columns] = data[num_columns].fillna(data[num_columns].mean())

    #drop columns with very large missing values 
    drop_columns = ['cat6','cat8']

    cat_columns.remove('cat6')
    cat_columns.remove('cat8')

    data = data.drop(columns = drop_columns)

    #handling missing values in categorical variables 
    data[cat_columns] = data[cat_columns].fillna("NA")

    #label encoding 
    cat_datatype_columns = data.select_dtypes(['object']).columns

    for column in cat_datatype_columns:
        data[column] = data[column].astype('category')

    data[cat_datatype_columns] = data[cat_datatype_columns].apply(lambda x:x.cat.codes)

    #one hot encoding
    data = one_hot_encoding(data,der_categorical_columns)
    data = one_hot_encoding(data,cat_columns)

    #normalize numerical values
    data = normalize_values(data,num_columns)
    #data = normalize_values(data,der_columns)
    return data

def get_preprocessed_data(train_path,test_path):
    data , target , test_id = prepare_data(train_path,test_path)
    prep_data = preprocess_data(data)
    prep_train_data = prep_data[:num_train]
    prep_test_data = prep_data[num_train:num_train + num_test]
    return prep_train_data , prep_test_data , target , test_id 


def get_train_test_data(prep_train_data,target,prep_test_data,n_train,n_val,n_test):
    #Divide data into training , validation and test set
    # removing id column
    prep_train_data = prep_train_data.drop(columns = ['id'])
    prep_test_data = prep_test_data.drop(columns = ['id'])
    y_train = target
    x_train = prep_train_data.values
    N,M = x_train.shape
    indices = np.arange(N)
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]


    x_val = x_train[:n_val]
    y_val = y_train[:n_val]
    val_set = (x_val,y_val)


    x_train = x_train[n_val:n_val+n_train]
    y_train = y_train[n_val:n_val+n_train]
    train_set = (x_train,y_train)

    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)

    #including a random y_test so iterator is compatible
    y_test = np.zeros((n_test),dtype=int)
    x_test = prep_test_data.values
    x_test = x_test[800000:n_test+800000]
    return train_set,val_set,(x_test,y_test)
