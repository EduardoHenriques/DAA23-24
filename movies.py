import sklearn as skl
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import datetime
from datetime import date

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
    
    # transformar first_air e last_air_date na diferenca do dia de hoje com a data respetiva
    dataset['first_air_date'] = pd.to_datetime(dataset['first_air_date'], format = '%Y-%m-%d', errors='coerce')
    dataset['last_air_date'] = pd.to_datetime(dataset['last_air_date'], format = '%Y-%m-%d', errors='coerce')
    today = datetime.datetime.today()
    dataset['first_air_date'] = (today - dataset['first_air_date']).dt.days
    dataset['last_air_date'] = (today - dataset['last_air_date']).dt.days
    print(dataset['first_air_date'])
    print(dataset['last_air_date'])
    # converter coluna adult em int (0 ou 1)
    dataset['adult'] = dataset['adult'].astype(int)
    # label encoding da coluna status e type
    lb_make = LabelEncoder()
    data['status'] = lb_make.fit_transform(data['status'])
    data['type'] = lb_make.fit_transform(data['type'])
    print(data['status'])
    
    
    print(dataset.info())
# correlação, heat map, historgramas, estatísticas, etc... """
def analysis(dataset):
    dataset_nums = dataset.select_dtypes(include=['number'])
    fig = plt.figure(figsize=(10,10))
    correlation = dataset_nums.corr(method = 'pearson')
    sns.heatmap(correlation, linecolor='black', linewidths=0.5)
    # sns.pairplot(dataset)
    plt.show()
    #
    # TODO : label encoding p/ fazer a correlação c/ as variaveis categoricas transformadas em numericas
    #

# START
if __name__ == "__main__":
    if os.path.exists(DATAFILE_PATH):
        print("Reading Data...")
        data = pd.read_csv(DATAFILE_PATH)
        filter(data)
        analysis(data)
    else:
        print(f"Error: File '{DATAFILE_PATH}' does not exist.")
    
    
