# conda install openpyxl
from scipy import stats
import sklearn as skl
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import date
import category_encoders as ce ## Necessário installar category_encoders in conda

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score


from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz #Decision tree 
from sklearn.ensemble import RandomForestClassifier

# Caminho do dataset
DATAFILE_PATH = "dataset/movies.csv"

# remove colunas consideradas inuteis, e remover linhas nulas do dataset restante
def filter(dataset):
    dataset.drop(columns=['id','tagline','backdrop_path','homepage','overview','production_companies','production_countries','poster_path'],axis=1,inplace=True)
    dataset.dropna(subset=['first_air_date',
                          'last_air_date',
                          'genres','created_by',
                          'languages','networks','origin_country',
                          'spoken_languages'],inplace=True)
    dataset = dataset[(dataset['number_of_seasons']>0) & (dataset['number_of_seasons'] < 7)]
    # transformar first_air e last_air_date na diferenca do dia de hoje com a data respetiva
    dataset['first_air_date'] = pd.to_datetime(dataset['first_air_date'], format = '%Y-%m-%d', errors='coerce')
    dataset['last_air_date'] = pd.to_datetime(dataset['last_air_date'], format = '%Y-%m-%d', errors='coerce')
    today = datetime.datetime.today()
    # dataset['first_air_date'] = (today - dataset['first_air_date']).dt.days
    # dataset['last_air_date'] = (today - dataset['last_air_date']).dt.days
    dataset['days_aired'] = (dataset['last_air_date'] - dataset['first_air_date']).dt.days
    dataset.drop(columns=['first_air_date','last_air_date'])
    # converter coluna adult em int (0 ou 1)
    dataset['adult'] = dataset['adult'].astype(int)
    # label encoding da coluna status e type
    lb_make = LabelEncoder()
    data['status'] = lb_make.fit_transform(data['status'])
    data['type'] = lb_make.fit_transform(data['type'])
    data['spoken_languages'] = lb_make.fit_transform(data['spoken_languages'])
    data['genres'] = lb_make.fit_transform(data['genres'])
    data['origin_country'] = lb_make.fit_transform(data['origin_country'])
    data['languages'] = lb_make.fit_transform(data['languages'])
    data['networks'] = lb_make.fit_transform(data['networks'])
    # outliers
    # esta tecnica de remover outliers retira alguns valores do dataset que têm um desvio de
    # 3*(desvio padrao) ou mais. As rows baixaram de 23074 para 21675.
    dataset.info()
    print("AFTER")
    numeric_columns = dataset.select_dtypes(include=['number'])
    z_scores = np.abs(stats.zscore(numeric_columns))
    outlier_rows = (z_scores > 3).any(axis=1)
    dataset_no_outliers = dataset[~outlier_rows]
    dataset_no_outliers.info()
    return dataset_no_outliers
    
# correlação, heat map, historgramas, estatísticas, etc... """
def analysis(dataset):
    dataset_nums = dataset.select_dtypes(include=['number'])
    fig = plt.figure(figsize=(10,10))
    correlation = dataset_nums.corr(method = 'pearson')
    # TODO : label encoding p/ fazer a correlação c/ as variaveis categoricas transformadas em numericas
    #
    #encoder = ce.BinaryEnconder(cols =[''])

def int_scorer(trueVAL,predVAL):
    roundedVAL = np.round(predVAL).astype(int)
    mse = mean_squared_error(trueVAL, roundedVAL)
    return mse

def modelo1_LinearReg(dataset):
    parameter = 'number_of_seasons'
    no_folds = 100
    excel_PATH = 'resultados/resultados_linearReg.xlsx'
    dataset.describe()
    # >>>> MODELO
    dataset = dataset.select_dtypes(include=['number'])
    # custom_scorer = make_scorer(int_scorer, greater_is_better=False)
    X = dataset.drop(columns=[parameter])
    Y = dataset[parameter]
    normalizer = MinMaxScaler()
    X_scaled = normalizer.fit_transform(X)
    Y_scaled = normalizer.fit_transform(Y.values.reshape(-1, 1))
    lm = LinearRegression()
    scores = cross_val_score(lm,X_scaled,Y_scaled,cv=no_folds)
    # <<<< MODELO
    print(scores)
    print("Result: %0.2f accuracy with std_dev of %0.2f" % (np.mean(scores),np.std(scores)))
    result = {
        '#Folds': [no_folds],
        'Parâmetro': [parameter],
        'Precisão': [scores.mean()],
        'Desvio Padrão': [scores.std()]
    }   
    save_result(result, excel_PATH)
    
def modelo2_RandomTree(dataset):
    no_folds = 300
    parameter = 'number_of_seasons'
    excel_PATH = 'resultados/resultados_tree.xlsx'
    # <<<<< MODELO
    dataset = dataset.drop(columns=['name','original_language','original_name','created_by','last_air_date','first_air_date'])
    dataset.info()
    X = dataset.drop(columns=[parameter])
    Y = dataset[parameter]
    clf = DecisionTreeClassifier(random_state=2023)
    scores = cross_val_score(clf,X,Y,cv=no_folds)
    # >>>>> MODELO
    print(scores)
    print("Result: %0.2f accuracy with std_dev of %0.2f" % (scores.mean(),scores.std()))
    result = {
        '#Folds': [no_folds],
        'Parâmetro': [parameter],
        'Precisão': [scores.mean()],
        'Desvio Padrão': [scores.std()]
    }   
    save_result(result, excel_PATH)

def modelo3_RandomForest(dataset):
    no_folds = 250
    estimators = 10
    dataset = dataset.select_dtypes(include=['number'])
    parameter = 'number_of_seasons'
    excel_PATH = 'resultados/resultados_forest.xlsx'
    parameter = 'number_of_seasons'
    # <<<<< MODELO
    # dataset = dataset.drop(columns=['name','original_language','original_name','created_by','last_air_date','first_air_date'])
    X = dataset.drop(columns=[parameter])
    Y = dataset[parameter]
    clf = RandomForestClassifier(n_estimators=estimators)
    scores = cross_val_score(clf, X, Y, cv=no_folds)
    # >>>>>> MODELO
    print(scores)
    print("Result: %0.2f accuracy with std_dev of %0.2f" % (scores.mean(),scores.std()))
    pd.Series(scores).hist()
    result = {
        '#Folds': [no_folds],
        'Parâmetro': [parameter],
        '#Árvores': [estimators],
        'Precisão': [scores.mean()],
        'Desvio Padrão': [scores.std()]
    }   
    save_result(result, excel_PATH)

def save_result(result, file_path):
    if not os.path.exists(file_path):
        df = pd.DataFrame(result)
        df.to_excel(file_path,index=False,engine='openpyxl')
    else:
        history = pd.read_excel(file_path, engine='openpyxl')
        results = pd.concat([history,pd.DataFrame(result)],axis=0, ignore_index=True)
        results.to_excel(file_path, index=False, engine='openpyxl')  

# START
if __name__ == "__main__":
    if os.path.exists(DATAFILE_PATH):
        print("Reading Data...")
        data = pd.read_csv(DATAFILE_PATH)
        data = filter(data)
        print("\n\n\nAFTER FILTERING...\n\n\n")
        modelo1_LinearReg(data)
        modelo3_RandomForest(data)
    else:
        print(f"Error: File '{DATAFILE_PATH}' does not exist.")
    
