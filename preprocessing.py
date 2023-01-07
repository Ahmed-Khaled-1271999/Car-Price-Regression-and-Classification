import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Car Info Parsing
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
    
def preprcessing_flow(data):
    
    data = data.drop(columns=['car_id'], axis='column')
    
    # تفكيك خاصية معلومات السيارة إلي مكونتها الثلاث: طراز و الشركة المُصّنعة و سنة الإنتاج
    data['car-model']           = data['car-info'].apply(lambda str_pattern: extract_car_info_feature_x(str_pattern, x='model'))
    data['car-manufacturer']    = data['car-info'].apply(lambda str_pattern: extract_car_info_feature_x(str_pattern, x='manufacturer'))
    data['car-production_year'] = data['car-info'].apply(lambda str_pattern: extract_car_info_feature_x(str_pattern, x='production_year'))
    data = data.drop(columns=['car-info'])
    
    # PETROL is petrol!... تسوية الألفاظ المؤدية لذات المعني 
    data['fuel_type'] = data['fuel_type'].apply(str.lower)
    
    # Drop Choice <Make Decision>:: Why not "Imputation" ???
    data = data.dropna(axis=0, how='any')
    
    # التعامل مع المتغيرات الفئوية بأن كل فئة لكل متغير هي خاصية مستقلة بذاتها يتعلمها النموذج
    # < Decision is made >: Are you sure you want to one-hot encode years ?????
    data = pd.get_dummies(data)
    
    # Standardize Numerical Feature
    ss = StandardScaler()
    scaled_features = ss.fit_transform(data[["mileage(kilometers)", "volume(cm3)"]])
    data[["mileage(kilometers)", "volume(cm3)"]] = scaled_features.tolist()
    
    pca = PCA()
    data = pca.fit_transform(data)
    
    return data

        
