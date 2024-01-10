# must install openpyxl in conda envoirnment to write the resulsts to an excel file
from menu import run_menu
from scipy import stats
import sklearn as skl
import os, datetime, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import category_encoders as ce  # Must install category_encoders in conda envoirnment
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_selection import f_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge   # Linear Regression
from sklearn.tree import DecisionTreeClassifier, export_graphviz #Decision tree 
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestClassifier # Random Forest
import tensorflow as tf                               # |
from tensorflow import keras                          # |
from tensorflow.keras.models import Sequential        # |  
from tensorflow.keras.layers import Dense             # |
from scikeras.wrappers import KerasRegressor          # |  
from scikeras.wrappers import KerasClassifier         # |
from keras.utils import to_categorical
from keras.utils import np_utils
#Metrics Imports
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

pd.set_option('mode.chained_assignment', None)

# Dataset Path
DATAFILE_PATH = "dataset/movies.csv"

# remove useless columns, and apply data processing
def filter(dataset):
    print('-'*60 + "\nORIGINAL DATESET\n" + '-'*60)
    # DROP USELESS COLUMNS/INFO, REMOVE MISSING VALUES
    dataset.info()
    dataset.drop(columns=['id','name','created_by','original_name','tagline','backdrop_path','homepage','overview','production_companies','production_countries','poster_path'],axis=1,inplace=True)
    #dataset.dropna(subset=['first_air_date',
    #                      'last_air_date',
    #                      'genres',
    #                      'languages','networks','origin_country',
    #                      'spoken_languages'],inplace=True)
    
    for col in ['first_air_date','last_air_date','genres','languages','networks','origin_country','spoken_languages']:
        dataset[col + '_missing'] = dataset[col].isnull().astype(int)
    # DAYS_AIRED ADDED, REMOVED DATES
    dataset['first_air_date'] = pd.to_datetime(dataset['first_air_date'], format = '%Y-%m-%d', errors='coerce')
    dataset['last_air_date'] = pd.to_datetime(dataset['last_air_date'], format = '%Y-%m-%d', errors='coerce')
    dataset['days_aired'] = (dataset['last_air_date'] - dataset['first_air_date']).dt.days
    dataset['days_aired'] = dataset['days_aired'][dataset['days_aired'] >= 0]
    dataset['days_aired'] = dataset['days_aired'].fillna(1)
    dataset.drop(columns=['last_air_date','first_air_date'],axis=1,inplace=True)
    # ADULT AS BOOL
    dataset['adult'] = dataset['adult'].astype(int)
    dataset['in_production'] = dataset['in_production'].astype(int)
    # LABEL ENCODING
    lb_make = LabelEncoder()
    columns_to_encode = ['status', 'type', 'spoken_languages', 'genres', 
                     'origin_country', 'languages', 'networks', 
                     'original_language']
    dataset[columns_to_encode] = dataset[columns_to_encode].apply(lambda col: lb_make.fit_transform(col))
    # REMOVE OUTLIERS
    # not sure about how good this technique is, but ~3000 rows were removed.
    # Z-score(number of sdt deviations from the mean) must be >3 to remove a row.
    numeric_columns = dataset.select_dtypes(include=['number']).columns
    z_scores = np.abs(stats.zscore(dataset[numeric_columns]))
    outlier_rows = (z_scores > 10).any(axis=1)
    dataset_no_outliers = dataset[~outlier_rows]
    print('-'*60 + "\nFILTERED DATASET\n" + '-'*60)
    dataset_no_outliers.info() 
    #dataset_no_outliers = dataset_no_outliers[(dataset_no_outliers['number_of_seasons']>0) & (dataset_no_outliers['number_of_seasons'] <= 7)]
    dataset_no_outliers['number_of_seasons'] = np.where(dataset_no_outliers['number_of_seasons'] >= 6, '6', dataset_no_outliers['number_of_seasons'])
    #dataset_no_outliers['number_of_seasons'] = pd.to_numeric(dataset_no_outliers['number_of_seasons'], errors='coerce').astype('Int64')
    # REMOVE PILOT EPISODES
    dataset_no_outliers = dataset_no_outliers[dataset_no_outliers['days_aired'] > 0]
    # RESULTS
    dataset_no_outliers.info() 
    return dataset_no_outliers
    
# correlação, heat map, historgramas, estatísticas, etc... """
def analysis(dataset):
    dataset.describe()
    #
    fig = plt.figure(figsize=(13, 13))
    correlation = dataset.corr(method = 'pearson')
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.savefig('corr.png')
    plt.show()
    #
    fig = plt.figure(figsize=(10, 10))
    sns.stripplot(x='number_of_seasons', y='days_aired', data=dataset.sort_values(by='number_of_seasons'), jitter=True, alpha=0.7)
    plt.xlabel('Nº Seasons')
    plt.ylabel('Days Aired')
    plt.savefig('seasons_vs_days.png')
    plt.show()
    #
    fig = plt.figure(figsize=(10, 10))
    incidents_count = dataset['number_of_seasons'].value_counts().sort_index()
    sns.set(style="darkgrid")
    sns.barplot(x=incidents_count.index, y=incidents_count.values)
    plt.title("Season Nº frequency")
    plt.xlabel("Number", fontsize=12)
    plt.ylabel("Ammount", fontsize=12)
    plt.savefig('season_amm.png')
    plt.show()
    #
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(dataset['popularity'], dataset['vote_average'])
    plt.xlabel('Popularity')
    plt.ylabel('Vote Average')
    plt.title('Scatter Plot: Popularity vs Vote average')
    plt.savefig('popularit_vs_voteAVG.png')
    plt.show()
    #
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(dataset['days_aired'], dataset['vote_average'])
    plt.xlabel('Days Aired')
    plt.ylabel('Vote Average')
    plt.title('Scatter Plot: Days Aired vs Vote average')
    plt.savefig('days_vs_voteAVG.png')
    plt.show()
    #
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(dataset['number_of_episodes'], dataset['days_aired'])
    plt.xlabel('Episodes')
    plt.ylabel('Days Aired')
    plt.title('Scatter Plot: Episodes vs Days Aired')
    plt.savefig('episodes_vs_days.png')
    plt.show()
    
#
# Linear Regression
#
def modelo1_LinearReg(dataset):
    lb_make = LabelEncoder()
    print("\n\n\nLINEAR REGRESSION\n\n\n")
    dataset.info()
    ###
    parameter = 'number_of_seasons'
    no_folds = 125
    excel_PATH = 'resultados/resultados_linearReg.xlsx'
    dataset.info()
    # >>>> MODEL
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
    MAE = mean_absolute_error(scores, [0] * len(scores))
    print("Result: %0.2f MSE with MAE of %0.2f" % (MSE,MAE))
    result = {
        '#Folds': [no_folds],
        'Parâmetro': [parameter],
        'MSE': [MSE],
        'MAE': [MAE]
    }   
    save_result(result, excel_PATH)

#
# Polynomial Regression
#
def modelo1_2_PolyReg(dataset):
	excel_PATH = 'resultados/resultados_PolyReg.xlsx'
	parameter = 'number_of_seasons'
	no_folds = 80
	degree = 3

	print("POLYNOMIAL REGRESSION MODEL SETUP")
	X = dataset.drop(columns=[parameter])
	Y = dataset[parameter]
	normalizer = MinMaxScaler()
	X_scaled = normalizer.fit_transform(X)
	Y_scaled = normalizer.fit_transform(Y.values.reshape(-1, 1))

	pl = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.25))
	selector = SelectKBest(score_func=f_regression, k=14)
	X_new = selector.fit_transform(X_scaled, Y_scaled.ravel())

	scores = cross_val_score(pl, X_new, Y_scaled.ravel(), cv=no_folds)
	print(scores)
	MSE = mean_squared_error(scores, [0] * len(scores))
	MAE = mean_absolute_error(scores, [0] * len(scores))
	print("Result: %0.2f MSE with MAE of %0.2f" % (MSE, MAE))
	result = {
		'#Folds': [no_folds],
		'Parâmetro': [parameter],
		'Degree': [degree],
		'MSE': [MSE],
		'MAE': [MAE],
	}
	save_result(result, excel_PATH)




# 
#  Decision Tree
#   
def modelo2_DecisionTree(dataset):
    no_folds = 50
    parameter = 'number_of_seasons'
    excel_PATH = 'resultados/resultados_tree.xlsx'
    
    X = dataset.drop(columns=[parameter])
    Y = dataset[parameter]
    clf = DecisionTreeClassifier(random_state=2023,criterion='gini',max_depth=10,min_samples_leaf=2,min_samples_split=10,splitter='best')
    scores = cross_val_score(clf,X,Y,cv=no_folds)
    
    print(scores)
    print("Result: %0.2f accuracy with std_dev of %0.2f" % (scores.mean(),scores.std()))
    result = {
        '#Folds': [no_folds],
        'Parâmetro': [parameter],
        'Precisão': [scores.mean()],
        'Desvio Padrão': [scores.std()]
    }   
    save_result(result, excel_PATH)
    #griddy(X,Y)


#
# Random Forest
#
def modelo3_RandomForest(dataset):
    no_folds = 50
    estimators = 100
    parameter = 'number_of_seasons'
    excel_PATH = 'resultados/resultados_forest.xlsx'
    # dataset = dataset.drop(columns=['name','original_language','original_name','created_by','last_air_date','first_air_date'])
    X = dataset.drop(columns=[parameter])
    Y = dataset[parameter]
    clf = RandomForestClassifier(n_estimators=estimators, criterion='gini', max_depth=30, min_samples_leaf=1, min_samples_split=5)
    scores = cross_val_score(clf, X, Y, cv=no_folds)
    
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

#
# MLP - Regression
#
def modelo4_MLP(dataset):
    excel_PATH = 'resultados/resultados_MLP.xlsx'
    parameter = 'number_of_seasons'
    activation = 'relu'
    rate = 0.01
    
    print("PERCEPTRON MODEL SETUP")
    my_model = build_model(activation,rate)
    # hyperparameter tuning
    optimizer = ['SGD','RMSprop','Adagrad']  
    param_grid = dict(optimizer=optimizer)
    kf = KFold(n_splits=5, shuffle=True,random_state=2021)
    model = KerasRegressor(model = my_model, batch_size=32, validation_split=0.2, epochs=20)
    y = dataset[parameter]
    x = dataset.drop(columns=parameter)
    normalizer = MinMaxScaler()
    X_scaled = normalizer.fit_transform(x)
    Y_scaled = normalizer.fit_transform(y.values.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,Y_scaled,test_size=0.2,random_state=101)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    
    print(f"MAE: {metrics.mean_absolute_error(y_test, predictions)}\n MSE: {metrics.mean_squared_error(y_test, predictions)}")
    result = {
        'Parâmetro': [parameter],
        'Activation': [activation],
        'Rate': [rate],
        'MSE': [round(metrics.mean_squared_error(y_test, predictions),2)],
        'MAE': [round(metrics.mean_absolute_error(y_test, predictions),2)]
    }
    save_result(result, excel_PATH)

#
# MLP - Classification
#
def modelo5_MLP_Classify(dataset):
    excel_PATH = 'resultados/resultados_MLP_Classification.xlsx'
    parameter = 'number_of_seasons'
    activation = 'relu'
    rate = 0.01
    num_classes = len(dataset[parameter].unique())  # Number of unique classes in the target variable

    print("PERCEPTRON MODEL SETUP")
    my_model = build_model_classification(activation, rate, num_classes)
    # hyperparameter tuning
    model = KerasClassifier(build_fn = my_model, batch_size=32, validation_split=0.2, epochs=20)
    y = dataset[parameter]
    x = dataset.drop(columns=parameter)
    normalizer = MinMaxScaler()
    X_scaled = normalizer.fit_transform(x)

    # Encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    # Convert integers to dummy variables 
    Y_scaled = np_utils.to_categorical(encoded_Y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled,Y_scaled,test_size=0.2,random_state=101)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    
    print(f"Accuracy: {metrics.accuracy_score(y_test, predictions)}")
    result = {
        'Parâmetro': [parameter],
        'Activation': [activation],
        'Rate': [rate],
        'Accuracy': [round(metrics.accuracy_score(y_test, predictions),2)]
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
    print(f"Model result saved in:{file_path}")

# builds a basic MLP model
def build_model(activation, learning_rate):
    model = Sequential() 
    model.add(Dense(32, input_dim = 23, activation = activation))
    model.add(Dense(16, activation = activation))
    model.add(Dense(8, activation = activation))
    model.add(Dense(4, activation = activation))
    model.add(Dense(1, activation = activation))
    model.compile(
        loss = 'mae',
        optimizer = tf.optimizers.Adam(learning_rate),
        metrics = ['mae','mse']
    )
    return model


def build_model_classification(activation, learning_rate, num_classes):
    model = Sequential() 
    model.add(Dense(32, input_dim = 23, activation = activation))
    model.add(Dense(16, activation = activation))
    model.add(Dense(8, activation = activation))
    model.add(Dense(num_classes, activation = 'softmax'))  

    model.compile(loss='categorical_crossentropy',  
                  optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    return model


def griddy(x,y):
    print("ENTRERING GRID SEARCH")
    param_grid = {
        'criterion': ["gini", "entropy"],
        'splitter': ["best", "random"],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    # Assuming your model is named 'model'
    #print("Feature names during training:", energia_total.get_params()['features'])
    #print("Feature names at prediction time:", energia23.columns.tolist())
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)
    grid = GridSearchCV(DecisionTreeClassifier(random_state=213123), param_grid, refit=True, verbose=3)
    grid.fit(x_train,y_train)
    print(grid.best_params_)


    


# START
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
    if os.path.exists(DATAFILE_PATH):
        print("Reading Data...")
        data = pd.read_csv(DATAFILE_PATH)
        data = filter(data)
        run_menu(modelo1_LinearReg, modelo1_2_PolyReg, modelo2_DecisionTree, modelo3_RandomForest, modelo4_MLP, modelo5_MLP_Classify, data)
        
    else:
        print(f"Error: File '{DATAFILE_PATH}' does not exist.")
    

