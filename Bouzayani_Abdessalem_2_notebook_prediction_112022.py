#!/usr/bin/env python
# coding: utf-8

# # Notebook de modélisation de la consommation totale d'énergie

# In[1]:



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
import matplotlib as mpl
import sys
import IPython as ip
import sklearn
import datetime
import re
import missingno as msno
from termcolor import colored

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Binarizer

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV,cross_validate
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,mean_squared_log_error

import time

from matplotlib.ticker import AutoMinorLocator
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')


# # 1. Importation  des données nettoyées

# In[2]:


debut_notebook = time.time()


# In[3]:


data = pd.read_csv('data_energy_nettoye.csv')


# In[4]:


#grader une copie du dataframe
df_energy=data.copy()


# In[5]:


data.sample(5)


# # 2. Modélisation consommation d'énergie

# In[6]:



# Fonction pour diviser le jeu de données en jeu d'entrainement et jeu de test
def partition_train_test(data,target,test_size):
    X = data.copy()
    X = X.drop(target, 1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size=test_size)
    return(X_train, X_test, y_train, y_test)


# In[7]:



# Fonction pour afficher les résultats sous forme d'un nuage de points
def scatter_plot(x, y, title, xaxis, yaxis, size, c_scale):
    trace = go.Scatter(
    x = x,
    y = y,
    mode = 'markers',
    marker = dict(color = y, size = size, showscale = True, colorscale = c_scale))
    layout = go.Layout(hovermode= 'closest', title = title, xaxis = dict(title = xaxis), yaxis = dict(title = yaxis))
    fig = go.Figure(data = [trace], layout = layout)
    return iplot(fig) 


# In[8]:


# Fonction pour calculer les métriques après entrainement et prédiction
def performances_model(var_model,nom_model):
    

    # Début d'exécution
    start_time = time.time()
    # Apprentissage du model
    var_model.fit(X_train, y_train)
    # Prédiction
    y_pred = var_model.predict(X_test)
    # Fin d'exécution
    end_time = time.time()

    #Calcul des métriques
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
  
    temp_execution = end_time - start_time

    # Création d'un dataframe des metriques calculées
    df_metrics = pd.DataFrame({
         'Modèle': [nom_model],
         'R2': [r2],
         'MSE': [mse],
         'RMSE': [rmse],
         'MAE': [mae],
         'Durée_exéc': [temp_execution]  })


    return (df_metrics,y_pred)


# In[9]:


# fonction pour afficher les résultats de prédiction des différents modèles et trouver le meilleur
def calcul_afichage_resultats(models,nom_models):
    df_performances = pd.DataFrame(columns=['Modèle','R2', 'MSE', 'RMSE', 'MAE', 'Durée_exéc' ])

    for i in range(len(models)):
        
        df_metrics_model = performances_model(models[i],nom_models[i])[0]
        df_performances = pd.concat([df_performances,df_metrics_model], ignore_index = True)
        df_performances1 = df_performances.sort_values(by = ['R2', 'RMSE'],ascending = [False, True])
        df_performances1 = df_performances1.reset_index(drop = True)

    print(df_performances1)
    # Graphique de comparaison des modèles suivant la métrique R2
    scatter_plot(df_performances['Modèle'], df_performances['R2'], ' Comparaison entre les modèles suivant R2', 'Models','Score R2', 30, 'Rainbow')
    # Graphique de comparaison des modèles suivant la métrique RMSE
    scatter_plot(df_performances['Modèle'], df_performances['RMSE'], ' Comparaison entre les modèles suivant RMSE', 'Models','Score RMSE', 30, 'Rainbow')
  
    print('Le meilleur modèle suivant la métrique R2 :',df_performances1.iloc[0,0])
    print('Le meilleur modèle suivant la métrique RMSE :',df_performances.loc[df_performances['RMSE'].idxmin(),'Modèle'])
  
    print('Le  modèle le plus rapide :',df_performances.loc[df_performances['Durée_exéc'].idxmin(),'Modèle'])

    return df_performances


# ## Premier modèle de régression (naif)
# 

# In[10]:


dummy_mean = DummyRegressor(strategy = 'mean')
dummy_median = DummyRegressor(strategy = 'median')
dummy_quantile = DummyRegressor(strategy='quantile', quantile=.75)
X_train, X_test, y_train, y_test = partition_train_test(data,'SiteEnergyUse_kBtu',0.2)
models = [ dummy_mean,dummy_median,dummy_quantile]
nom_models = [ 'dummy_mean','dummy_median','dummy_quantile']
df_performances_sans_normalisation = calcul_afichage_resultats(models,nom_models)


# Ces modèles (naifs) de régression donnent de mauvaises résultats, je vais utiliser d'autres modèles

# ## Plusieurs modèles de régression sans la variable 'ENERGYSTARScore'

# **Prédiction sans normalisation**

# In[11]:


data = data.drop('ENERGYSTARScore',1)


# In[12]:


X_train, X_test, y_train, y_test = partition_train_test(data,'SiteEnergyUse_kBtu',0.2)


# In[13]:


# Définition des modèles de régression (initialisation par défaut des paramètres)
seed = 42
jobs=-1
def initialisation_par_defaut():
    
    linear = LinearRegression(n_jobs = jobs)
    lasso = Lasso(random_state = seed)
    ridge = Ridge(random_state = seed)
    kr = KernelRidge()
    elnt = ElasticNet(random_state = seed)
    dt = DecisionTreeRegressor(random_state = seed)
    svm = SVR()
    knn = KNeighborsRegressor(n_jobs = jobs)
    rf =  RandomForestRegressor(n_jobs = jobs, random_state = seed)
    et = ExtraTreesRegressor(n_jobs = jobs, random_state = seed)
    ab = AdaBoostRegressor(random_state = seed)
    gb = GradientBoostingRegressor(random_state = seed)
    xgb = XGBRegressor(verbosity = 0,random_state = seed, n_jobs = jobs)
    lgb = LGBMRegressor(random_state = seed, n_jobs = jobs)
    models = [ linear, lasso, ridge, kr,elnt, dt,svm, knn, rf, et, ab, gb, xgb, lgb]
    return models


# In[14]:


nom_models = [ 'linear', 'lasso', 'ridge','kr', 'elnt', 'dt','svm', 'knn', 'rf', 'et', 'ab', 'gb', 'xgb', 'lgb']
models = initialisation_par_defaut()
df_performances_sans_normalisation = calcul_afichage_resultats(models,nom_models)


# **Prédiction avec normalisation**

# In[15]:


colonnes_a_normaliser = ['Latitude',
 'Longitude',
 'NumberofBuildings',
 'NumberofFloors',
 'PropertyGFATotal',
 'PropertyGFAParking',
 'LargestPropertyUseTypeGFA',
 'SecondLargestPropertyUseTypeGFA',
 'ThirdLargestPropertyUseTypeGFA',
 'SiteEnergyUse_kBtu',
 'Building_Age',
 'Rate_Parking',
 'Rate_LargestPropertyUseType',
 'Rate_SecondLargestPropertyUseType',
 'Rate_ThirdLargestPropertyUseType']


# In[16]:


data_a_normaliser = data[colonnes_a_normaliser].copy()
scaler = preprocessing.RobustScaler()

data_a_normaliser = scaler.fit_transform(data_a_normaliser)
data_a_normaliser = pd.DataFrame(data_a_normaliser, 
                                columns=colonnes_a_normaliser,
                                index =data.index.to_list())
for colonne in colonnes_a_normaliser:
    data[colonne] = data_a_normaliser[colonne]


# In[17]:


X_train, X_test, y_train, y_test = partition_train_test(data,'SiteEnergyUse_kBtu',0.2)


# In[18]:


nom_models = [ 'linear', 'lasso', 'ridge','kr', 'elnt', 'dt','svm', 'knn', 'rf', 'et', 'ab', 'gb', 'xgb', 'lgb']
models = initialisation_par_defaut()
df_performances_avec_normalisation = calcul_afichage_resultats(models,nom_models)


# In[19]:


# Fonction pour afficher les résultats de comparaison
def afficher_resultats_comparaison(df1,methode1,df2,methode2,metrique):
  
  df = pd.DataFrame(columns =['Modèle','methode1','methode2'])
  df['Modèle'] = df1['Modèle']
  df[methode1] = df1[metrique]
  df[methode2] = df2[metrique]

  #df = pd.DataFrame([df1['Modèle'],df1[metrique],df2[metrique]])
  ax = df.plot(x="Modèle", y=[methode1, methode2], kind="bar", rot=0,title="comparaison des deux méthodes suivant le score "+metrique,figsize=(12,5))

 


# In[20]:


# Fonction  de comparaison des résultats

def comparaison_resultats_prediction(df1,methode1,df2,methode2):
    dfa=df1.copy()
    dfb=df2.copy()
    dfa.set_index('Modèle',inplace=True)
    dfb.set_index('Modèle',inplace=True)
    df = dfa[['R2','RMSE']].subtract(dfb[['R2','RMSE']], axis = 1)
    nb_r2=len(df.loc[df['R2'] > 0])
    #nb_rmse=len(df.loc[df['RMSE'] > 0])
    #print(df)
    if nb_r2/df.shape[0]>=0.5:
        
        print('Pour la majorité des modèles, les résultats avec la transformation :',methode1,'sont meilleurs selon le score R2')
    else:
        print('Pour la majorité des modèles, les résultats avec la transformation :',methode2,'sont meilleurs selon le score R2')
    afficher_resultats_comparaison(df1,methode1,df2,methode2,"R2")
   


# In[21]:


comparaison_resultats_prediction(df_performances_avec_normalisation,'avec normalisation',df_performances_sans_normalisation,'sans normalisation')


# **Prédiction avec passage au Log**

# In[22]:


data = df_energy.copy()
colonnes_log = ['PropertyGFATotal',
                'PropertyGFAParking',
                'LargestPropertyUseTypeGFA',
                'SecondLargestPropertyUseTypeGFA',
                'ThirdLargestPropertyUseTypeGFA',
                'SiteEnergyUse_kBtu',
                ]
data_log = data[colonnes_log].copy()

for colonne in colonnes_log:
    
    data_log[colonne] = np.log(1+data[colonne])
    plt.figure(figsize=(15, 5))
    plt.subplot( 1,2 ,1)
    sns.histplot(data[colonne], kde=True, color='blue')
    plt.title(f'Distribution de  {colonne}')
    plt.subplot(1,2,2 )
    sns.histplot(data_log[colonne], kde=True, color='blue')
    plt.title(f'Distributions de  {colonne} après passage au Log')        
    plt.show()
    data[colonne] = data_log[colonne]


# In[23]:


data_log = data.copy()
data = data.drop('ENERGYSTARScore',1)
X_train, X_test, y_train, y_test = partition_train_test(data,'SiteEnergyUse_kBtu',0.2)
nom_models = [ 'linear', 'lasso', 'ridge','kr', 'elnt', 'dt','svm', 'knn', 'rf', 'et', 'ab', 'gb', 'xgb', 'lgb']
models = initialisation_par_defaut()
df_performances_avec_log = calcul_afichage_resultats(models,nom_models)


# In[24]:


comparaison_resultats_prediction(df_performances_avec_log,'avec passage au log',df_performances_avec_normalisation,'sans passage au log')


# Dans la suite, je vais travailler avec la  transformation : passage au log

# In[25]:


data = data_log.copy()
data = data.drop('ENERGYSTARScore',1)
X_train, X_test, y_train, y_test = partition_train_test(data,'SiteEnergyUse_kBtu',0.2)


# **Cross validation** 
Pour s'assurer de la stabilité des modèles et éviter le sur-apprentissage, on doit utiliser une validation croisée
# In[26]:


def cross_validation_score_r2_mse(var_model,nom_model):
      
    # cross validation
    scoring = ['r2', 'neg_mean_squared_error']
    scores = cross_validate(var_model, X_train, y_train, cv=10,scoring=scoring, return_train_score=True)

  
    # Création d'un dataframe avec les metriques moyennes
    df_metrics = pd.DataFrame({
         'Modèle': [nom_model],
         'Test_R2_CV': [scores['test_r2'].mean()],
         'Test_MSE_CV': [-(scores['test_neg_mean_squared_error'].mean())],
         'Train_R2_CV': [scores['train_r2'].mean()],
         'Train_MSE_CV': [-(scores['train_neg_mean_squared_error'].mean())]  })

    
    return df_metrics


# In[27]:


models = initialisation_par_defaut()
nom_models = [ 'linear', 'lasso', 'ridge','kr', 'elnt', 'dt','svm', 'knn', 'rf', 'et', 'ab', 'gb', 'xgb', 'lgb']
df_performances = pd.DataFrame(columns=['Modèle','Test_R2_CV', 'Test_MSE_CV', 'Train_R2_CV', 'Train_MSE_CV'])

for i in range(len(models)):
    df_metrics_model = cross_validation_score_r2_mse(models[i],nom_models[i])
    df_performances = pd.concat([df_performances,df_metrics_model], ignore_index = True)

print(df_performances)
# Graphique de comparaison des modèles suivant la métrique R2
scatter_plot(df_performances['Modèle'], df_performances['Test_R2_CV'], ' Comparaison entre les modèles suivant R2', 'Models','Score R2', 30, 'Rainbow')
# Graphique de comparaison des modèles suivant la métrique RMSE
scatter_plot(df_performances['Modèle'], df_performances['Test_MSE_CV'], ' Comparaison entre les modèles suivant MSE', 'Models','Score RMSE', 30, 'Rainbow')

print('Le meilleur modèle suivant la métrique R2 :',df_performances.loc[df_performances['Test_R2_CV'].idxmax(),'Modèle'])
print('Le meilleur modèle suivant la métrique MSE :',df_performances.loc[df_performances['Test_MSE_CV'].idxmin(),'Modèle'])


# **Recherhe des hyperparametres**
Je vais chercher les hyperparametres des 3 meilleurs modèles qui ont attiré mon attention (rf, gb et lgb)
# In[28]:


def grid_search_cv(model, params):
    global best_params, best_score
    
   
    grid_search = GridSearchCV(estimator = model, param_grid = params, cv = 10, verbose = 1,
                            scoring = 'neg_mean_squared_error', n_jobs = -1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_ 
    best_score = np.sqrt(-1*(grid_search.best_score_))
   
    return best_params, best_score


# In[29]:


def recherche_hyperparameters(models_hyperparameters):
    best_hyperparameters = {}
    for model,params in models_hyperparameters.items():
        start_time = time.time()
        best_params, best_score = grid_search_cv(model, params)
        end_time = time.time()
        best_hyperparameters.update({model:best_params})
        print('Model:{} a best params:{} & best_score:{:0.5f} et s\'est exécuté dans {} secondes' .format(model,best_params, best_score,(end_time-start_time)))
    return(best_hyperparameters)


# In[30]:


rf =  RandomForestRegressor(n_jobs = jobs, random_state = seed)
gb = GradientBoostingRegressor(random_state = seed)
lgb = LGBMRegressor(random_state = seed, n_jobs = jobs)


# In[31]:


# Grille de recherche
models_hyperparameters = {  rf: {'max_depth': [10,50,100],  'n_estimators': [10, 50, 100],'min_samples_split' : [2, 5, 10],'min_samples_leaf' : [1, 3, 4]  },
                            gb: {'learning_rate': [0.01,0.1,0.5],'subsample' : [ 0.1,0.9,1],'n_estimators' : [100,500],'max_depth': [1,10],'min_samples_split' : [2, 5],'min_samples_leaf' : [1, 3]},
                            lgb: {'learning_rate': [0.01,0.05,0.1, 0.5,0.9],'n_estimators' : [100,500,1000],'max_depth': [1,3,5,10]  }
                                                          
                          }
                

best_hyperparameters = recherche_hyperparameters(models_hyperparameters)


# In[32]:


# Les meilleurs paramètres trouvés pour chaque modèle
best_hyperparameters


# In[33]:


# Initialiser les modèles avec leurs meilleurs hyperparameters
def Initialiser_hyperparameters(best_hyperparameters):
    rf_opt =  RandomForestRegressor(**best_hyperparameters[rf])
    gb_opt = GradientBoostingRegressor(**best_hyperparameters[gb])
    lgb_opt = LGBMRegressor(**best_hyperparameters[lgb])
    models = [ rf_opt, gb_opt , lgb_opt]

    return models


# In[34]:


# Calculer les résultats des models avec les meillleurs paramètres
nom_models = [ 'rf','gb', 'lgb']
models = Initialiser_hyperparameters(best_hyperparameters)
df_performances_sans_energystarscor = calcul_afichage_resultats(models,nom_models)
df_performances_par_defaut = df_performances_avec_normalisation.iloc[[8,11,13],:]
df_performances_par_defaut = df_performances_par_defaut.reset_index(drop=True)
comparaison_resultats_prediction(df_performances_sans_energystarscor,'avec best hyperparameters', df_performances_par_defaut,'par défaut')

Nous remarquons que l'utilisation des meilleuts hyperpatametres améliore les résultats pour les 3 modèles de régression
# In[35]:


# les 3 meilleurs modèles avec initialisation par défaut

df_performances_par_defaut


# In[36]:


# les 3 meilleurs modèles avec best parameters

df_performances_sans_energystarscor


# ## Régression avec  la variable 'ENERGYSTARScore'

# In[37]:


data['ENERGYSTARScore'] = df_energy['ENERGYSTARScore']


# In[38]:


#Imputation des valeurs manquantes de la variable 'ENERGYSTARScore' en utilisant la médiane
data['ENERGYSTARScore'].fillna(data.groupby('NumberofBuildings')['ENERGYSTARScore'].transform('median'), inplace = True)
data = data.dropna()


# In[39]:


data.isnull().mean().mean()


# In[40]:


X_train, X_test, y_train, y_test = partition_train_test(data,'SiteEnergyUse_kBtu',0.2)


# In[41]:


# Calculer les résultats des models avec les meillleurs paramètres

nom_models = ['rf',  'gb', 'lgb']
models = Initialiser_hyperparameters(best_hyperparameters)
df_performances_avec_energystarscor = calcul_afichage_resultats(models,nom_models)


# ## Utilité de la variable 'ENERGYSTARScore'

# In[42]:


comparaison_resultats_prediction(df_performances_avec_energystarscor,'avec ENERGYSTARScore',df_performances_sans_energystarscor,'sans ENERGYSTARScore')


# In[43]:


df_performances_sans_energystarscor


# In[44]:


df_performances_avec_energystarscor


# ## Conclusion

# Les 3 meilleurs modèles sont : 
#  - Gradient Boosting
#  - Light Gradient Boosting
#  - Random forest
#  
# Une légère amélioration des résultats avec la variable ENERGYSTARScore

# In[45]:


fin_notebook = time.time()


# In[46]:


print('Ce notebook prend ',(fin_notebook - debut_notebook)/60,' minutes pour s\'exécuter')


# In[ ]:




