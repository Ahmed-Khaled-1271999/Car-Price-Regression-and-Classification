import pandas as pd

def extract_car_info_feature_x(str_pattern, x):
    str_pattern.strip()
    str_pattern = str_pattern[1: len(str_pattern)-1]
    str_pattern = [pattern[1: len(pattern)-1] for pattern in str_pattern.split(',')] 
    if x == 'model':
        return str_pattern[0]
    elif x == 'manufacturer':
        return str_pattern[1]
    elif x == 'production_year':
        return str_pattern[2]
    else:
        raise Exception(f"I can not Extract this Feature ! \n ")

# Testing Set Catch Up
def catch_up_testing(X_testing):
    X_testing = X_testing.drop(columns=['car_id'], axis='column')
    # 
    X_testing['car-model']           = X_testing['car-info'].apply(lambda str_pattern: extract_car_info_feature_x(str_pattern, x='model'))
    X_testing['car-manufacturer']    = X_testing['car-info'].apply(lambda str_pattern: extract_car_info_feature_x(str_pattern, x='manufacturer'))
    X_testing['car-production_year'] = X_testing['car-info'].apply(lambda str_pattern: extract_car_info_feature_x(str_pattern, x='production_year'))
    X_testing = X_testing.drop(columns=['car-info'])
    # 
    X_testing['fuel_type'] = X_testing['fuel_type'].apply(str.lower)
    # 
    X_testing = X_testing.dropna(axis=0, how='any')
    
    return X_testing

def difference_reporting():
    pass

def ls_categories(df):
    categorical_columns = set(df.columns) - {'volume(cm3)', 'mileage(kilometers)'}
    for col in list(categorical_columns):
        print(f'Feature {col} has {len(list(df[col].unique()))} Unique Possible Values : {list(df[col].unique())}')
        print(f'{300*"="}')
        
def training_diff_testing(X_train, X_test):
    training_difference_testing = []
    # record for each single feature; what is present on training for that feature Does NOT exit in Testing
    categorical_columns = set(X_train.columns) - set([ 'Price Category', 'volume(cm3)', 'mileage(kilometers)'])    # SAME either we use Training or Testing Set
    for col in list(categorical_columns):
        if len(set(X_train[col].unique()).difference(set(X_test[col].unique()))) > 0:
            print(f'For Feature {col}; Training set DIFF Testing Set {len(set(X_train[col].unique()).difference(set(X_test[col].unique())))} Values : {set(X_train[col].unique()).difference(set(X_test[col].unique()))}')
            print(f'{300*"="}')
            training_difference_testing.extend(list(set(X_train[col].unique()).difference(set(X_test[col].unique()))))

def testing_diff_training(X_train, X_test):
    testing_difference_training = []
    # record for each single feature; what is present on testing for that feature Does NOT exit in training
    categorical_columns = set(X_train.columns) - set([ 'Price Category', 'volume(cm3)', 'mileage(kilometers)'])    # SAME either we use Training or Testing Set
    for col in list(categorical_columns):
        if len(set(X_test[col].unique()).difference(set(X_train[col].unique()))) > 0:
            print(f'For Feature {col}; Testing set DIFF Training Set {len(set(X_test[col].unique()).difference(set(X_train[col].unique()))) } Values : {set(X_test[col].unique()).difference(set(X_train[col].unique()))}')  
            print(f'{300*"="}')
            testing_difference_training.extend(list(set(X_test[col].unique()).difference(set(X_train[col].unique()))))

def mirroring(X_train, X_test):
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    
    training_difference_testing = list(set(X_train.columns).difference(set(X_test.columns)))
    testing_difference_training = list(set(X_test.columns).difference(set(X_train.columns)))
   
    
    # أسس للخاصية الحاضرة فى التيست و ليست حاضرة هنا
    for absent_tarining_feature in testing_difference_training:
        X_train[str(absent_tarining_feature)] = 0
    
    # أسس للخاصية الحاضرة فى التدريب و ليست حاضرة هنا
    for absent_testing_feature in training_difference_testing:
        X_test[str(absent_testing_feature)] = 0
        
    for col in list(X_train.columns):
        # print(f'X_testing_set {col} {X_test[col].dtype}, X_training_set {col} {X_train[col].dtype}')
        X_test[col].astype(X_train[col].dtype)
    
    # re-order the columns based on another dataframe 
    X_test = X_test[X_train.columns]

    return X_train, X_test
    