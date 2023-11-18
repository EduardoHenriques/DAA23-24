# must install openpyxl in conda envoirnment to write the resulsts to an excel file
from scipy import stats
import sklearn as skl
import os, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import category_encoders as ce  # Must install category_encoders in conda envoirnment
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, cross_val_score
# Model Imports
from sklearn.linear_model import LinearRegression   # Linear Regression
from sklearn.tree import DecisionTreeClassifier, export_graphviz #Decision tree 
from sklearn.ensemble import RandomForestClassifier # Random Forest
import tensorflow as tf                               # |
from tensorflow import keras                          # |
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor          # |  
#Metrics Imports
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

# Dataset Path
DATAFILE_PATH = "dataset/movies.csv"

# remove useless columns, and apply data processing
def filter(dataset):
    dataset.drop(columns=['id','tagline','backdrop_path','homepage','overview','production_companies','production_countries','poster_path'],axis=1,inplace=True)
    dataset.dropna(subset=['first_air_date',
                          'last_air_date',
                          'genres','created_by',
                          'languages','networks','origin_country',
                          'spoken_languages'],inplace=True)
    dataset = dataset[(dataset['number_of_seasons']>0) & (dataset['number_of_seasons'] < 7)]
    # turn first_air_date and last_air_date into days_aired
    dataset['first_air_date'] = pd.to_datetime(dataset['first_air_date'], format = '%Y-%m-%d', errors='coerce')
    dataset['last_air_date'] = pd.to_datetime(dataset['last_air_date'], format = '%Y-%m-%d', errors='coerce')
    today = datetime.datetime.today()
    dataset['days_aired'] = (dataset['last_air_date'] - dataset['first_air_date']).dt.days
    dataset['days_aired'] = dataset['days_aired'][dataset['days_aired'] >= 0]
    dataset.drop(columns=['first_air_date','last_air_date'])
    # adult goes from 0/1 rather than False/True
    dataset['adult'] = dataset['adult'].astype(int)
    # label encoding to a few attributes
    lb_make = LabelEncoder()
    data['status'] = lb_make.fit_transform(data['status'])
    data['type'] = lb_make.fit_transform(data['type'])
    data['spoken_languages'] = lb_make.fit_transform(data['spoken_languages'])
    data['genres'] = lb_make.fit_transform(data['genres'])
    data['origin_country'] = lb_make.fit_transform(data['origin_country'])
    data['languages'] = lb_make.fit_transform(data['languages'])
    data['networks'] = lb_make.fit_transform(data['networks'])
    # remove outliers
    # not sure about how good this technique is, but ~3000 rows were removed.
    # Z-score(number of sdt deviations from the mean) must be >3 to remove a row.
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
    dataset.describe()
    # dataset_nums = dataset.select_dtypes(include=['number'])
    fig = plt.figure(figsize=(10,10))
    #correlation = dataset_nums.corr(method = 'pearson')
    #sns.stripplot(x='number_of_seasons', y='days_aired', data=dataset, jitter=True, alpha=0.7)
    #sns.violinplot(x='number_of_seasons', y='days_aired', data=dataset)
    incidents_count = dataset['number_of_seasons'].value_counts()
    sns.set(style="darkgrid")
    sns.barplot(x=incidents_count.index, y=incidents_count.values)
    plt.title("Season Nº frequency")
    plt.xlabel("Number", fontsize=12)
    plt.ylabel("Ammount", fontsize=12)
    plt.show()
    
def int_scorer(trueVAL,predVAL):
    roundedVAL = np.round(predVAL).astype(int)
    mse = mean_squared_error(trueVAL, roundedVAL)
    return mse

def modelo1_LinearReg(dataset):
    lb_make = LabelEncoder()
    dataset['status'] = lb_make.fit_transform(data['status'])
    dataset['type'] = lb_make.fit_transform(data['type'])
    dataset['spoken_languages'] = lb_make.fit_transform(data['spoken_languages'])
    dataset['genres'] = lb_make.fit_transform(data['genres'])
    dataset['origin_country'] = lb_make.fit_transform(data['origin_country'])
    dataset['original_language'] = lb_make.fit_transform(data['original_language'])
    dataset['languages'] = lb_make.fit_transform(data['languages'])
    dataset['networks'] = lb_make.fit_transform(data['networks'])
    dataset['in_production'] = dataset['in_production'].astype(int)
    dataset.drop(columns=['name','original_name','created_by'],axis=1,inplace=True)
    print("\n\n\nLINEAR REGRESSION\n\n\n")
    dataset.info()
    ###
    parameter = 'number_of_seasons'
    no_folds = 100
    excel_PATH = 'resultados/resultados_linearReg.xlsx'
    dataset.info()
    # >>>> MODEL
    dataset = dataset.select_dtypes(include=['number'])
    X = dataset.drop(columns=[parameter])
    Y = dataset[parameter]
    normalizer = MinMaxScaler()
    X_scaled = normalizer.fit_transform(X)
    Y_scaled = normalizer.fit_transform(Y.values.reshape(-1, 1))
    lm = LinearRegression()
    scores = cross_val_score(lm,X_scaled,Y_scaled,cv=no_folds)
    # <<<< MODEL
    print(scores)
    MSE = mean_squared_error(scores, [0] * len(scores))
    MSA = mean_absolute_error(scores, [0] * len(scores))
    print("Result: %0.2f MSE with MSA of %0.2f" % (MSE,MSA))
    result = {
        '#Folds': [no_folds],
        'Parâmetro': [parameter],
        'MSE': [MSE],
        'MAE': [MSA]
    }   
    save_result(result, excel_PATH)
    
def modelo2_DecisionTree(dataset):
    no_folds = 300
    parameter = 'number_of_seasons'
    excel_PATH = 'resultados/resultados_tree.xlsx'
    # <<<<< MODEL
    dataset = dataset.drop(columns=['name','original_language','original_name','created_by','last_air_date','first_air_date'])
    dataset.info()
    X = dataset.drop(columns=[parameter])
    Y = dataset[parameter]
    clf = DecisionTreeClassifier(random_state=2023)
    scores = cross_val_score(clf,X,Y,cv=no_folds)
    # >>>>> MODEL
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
    no_folds = 10
    estimators = 500
    dataset = dataset.select_dtypes(include=['number'])
    parameter = 'number_of_seasons'
    excel_PATH = 'resultados/resultados_forest.xlsx'
    # <<<<< MODEL
    # dataset = dataset.drop(columns=['name','original_language','original_name','created_by','last_air_date','first_air_date'])
    X = dataset.drop(columns=[parameter])
    Y = dataset[parameter]
    clf = RandomForestClassifier(n_estimators=estimators)
    scores = cross_val_score(clf, X, Y, cv=no_folds)
    # >>>>>> MODEL
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

def modelo4_MLP(dataset):
    excel_PATH = 'resultados/resultados_MLP.xlsx'
    dataset = dataset = dataset.select_dtypes(include=['number'])
    parameter = 'number_of_seasons'
    print("PERCEPTRON MODEL SETUP")
    my_model = build_model()
    # hyperparameter tuning
    optimizer = ['SGD','RMSprop','Adagrad']  
    param_grid = dict(optimizer=optimizer)
    kf = KFold(n_splits=5, shuffle=True,random_state=2021)
    model = KerasRegressor(model = my_model, batch_size=32, validation_split=0.2, epochs=20)
    # split data
    y = dataset[parameter]
    X = dataset.drop(columns=parameter)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=101)
    grid_search = GridSearchCV(estimator=model, param_grid =param_grid, cv = kf, scoring='neg_mean_absolute_error', refit='True', verbose=10)
    grid_search.fit(X_train,y_train)
    print("Best %f  using %s is:"  % (grid_search.best_score_,grid_search.best_params_))
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    params = grid_search.cv_results_['params']
    
def save_result(result, file_path):
    if not os.path.exists(file_path):
        df = pd.DataFrame(result)
        df.to_excel(file_path,index=False,engine='openpyxl')
    else:
        history = pd.read_excel(file_path, engine='openpyxl')
        results = pd.concat([history,pd.DataFrame(result)],axis=0, ignore_index=True)
        results.to_excel(file_path, index=False, engine='openpyxl')  
    print(f"Model result saved in:{file_path}")

# builds a basic MLP model
def build_model(activation='relu', learning_rate=0.01):
    model = Sequential() 
    model.add(Dense(8, input_dim = 16, activation = activation))
    model.add(Dense(1, activation = activation))
    model.compile(
        loss = 'mae',
        optimizer = tf.optimizers.Adam(learning_rate),
        metrics = ['mae','mse']
    )
    return model

# START
if __name__ == "__main__":
    if os.path.exists(DATAFILE_PATH):
        print("Reading Data...")
        data = pd.read_csv(DATAFILE_PATH)
        data = filter(data)
        analysis(data)
        print("\n\n\nAFTER FILTERING...\n\n\n")
        #modelo1_LinearReg(data)
        #modelo4_MLP(data)
        # modelo3_RandomForest(data)
    else:
        print(f"Error: File '{DATAFILE_PATH}' does not exist.")
    

