#!/usr/bin/env python
# coding: utf-8

# # Projet 4

# **Énoncé**
# 
# Vous travaillez pour la ville de Seattle. Pour atteindre son objectif de ville neutre en émissions de carbone en 2050, votre équipe s’intéresse de près à la consommation et aux émissions des bâtiments non destinés à l’habitation.
# 
# **Les données**
# 
# Des relevés minutieux ont été effectués par les agents de la ville en 2016. Voici les [données](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/Data_Scientist_P4/2016_Building_Energy_Benchmarking.csv) et [leur source](https://data.seattle.gov/dataset/2016-Building-Energy-Benchmarking/2bpz-gwpy). Cependant, ces relevés sont coûteux à obtenir, et à partir de ceux déjà réalisés, vous voulez tenter de prédire les émissions de CO2 et la consommation totale d’énergie de bâtiments non destinés à l’habitation pour lesquels elles n’ont pas encore été mesurées.
# 
# **Mission**
# 
# Vous cherchez également à évaluer l’intérêt de l’"ENERGY STAR Score" pour la prédiction d’émissions, qui est fastidieux à calculer avec l’approche utilisée actuellement par votre équipe. Vous l'intégrerez dans la modélisation et jugerez de son intérêt.
# 
# Vous sortez tout juste d’une réunion de brief avec votre équipe. Voici un récapitulatif de votre mission :
# 
# Réaliser une courte analyse exploratoire.
# Tester différents modèles de prédiction afin de répondre au mieux à la problématique.
# Avant de quitter la salle de brief, Douglas, le project lead, vous donne quelques pistes et erreurs à éviter :
# 
# 
# 
# Douglas : L’objectif est de te passer des relevés de consommation annuels futurs (attention à la fuite de données). Nous ferons de toute façon pour tout nouveau bâtiment un premier relevé de référence la première année, donc rien ne t'interdit d’en déduire des variables structurelles aux bâtiments, par exemple la nature et proportions des sources d’énergie utilisées.. 
# 
# Fais bien attention au traitement des différentes variables, à la fois pour trouver de nouvelles informations (peut-on déduire des choses intéressantes d’une simple adresse ?) et optimiser les performances en appliquant des transformations simples aux variables (normalisation, passage au log, etc.).
# 
# Mets en place une évaluation rigoureuse des performances de la régression, et optimise les hyperparamètres et le choix d’algorithmes de ML à l’aide d’une validation croisée.
# 
# **Livrables attendus**
# 
# Un **notebook de l'analyse exploratoire** mis au propre et annoté.
# 
# Un **notebook pour chaque prédiction** (émissions de CO2 et consommation totale d’énergie) des différents tests de modèles mis au propre, dans lequel vous identifierez clairement le modèle final choisi.
# 
# Un **support de présentation** pour la soutenance (entre 15 et 25 slides).

# **But du projet :**
# 
# Le but du projet est de : prédire les émissions de CO2 **target 2: TotalGHGEmissions**
# 
# prédire la consommation totale d’énergie de bâtiments **target 1: SiteEnergyUse_kBtu**
# 

# # Notebook de nettoyage

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
from sklearn.ensemble import RandomForestRegressor

from yellowbrick.model_selection import FeatureImportances



from matplotlib.ticker import AutoMinorLocator
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')



# # 1. Importation et découverte des données 

# In[2]:


data = pd.read_csv('2016_Building_Energy_Benchmarking.csv')


# In[3]:


#grader une copie du dataframe
df_building=data.copy()


# In[4]:


data.head(5)


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


print('1 ------------------------------------------------------------------------')
print ('Il y a ', data.shape[0], 'lignes et ', data.shape[1],'colonnes dans la base ')
# données manquantes par colonnes
print('2 ------------------------------------------------------------------------')
print('Le nombre de données manquantes par colonnes : \n',data.isna().sum())
# données manquantes dans toute la base
print('3 ------------------------------------------------------------------------')
print('Le nombre total de données manquantes est : \n',data.isna().sum().sum())
# pourcentage des données manquantes 
print('4 ------------------------------------------------------------------------')
print('Le pourcentage des données manquantes est : \n',round(data.isna().mean().mean()*100,2),'%')
# nombre de doublons sur toutes les colonnes
if data.duplicated().unique():
    print ('Il y a ', data.duplicated().sum(), 'lignes dupliquées')
else:
    print('Il n\'y a pas de doublons dans cette base')


# In[8]:


# liste des colonnes
listCol = data.columns.tolist()
listCol


# In[9]:


# remplacer les caracteres '(','/' par '_' dans les noms de colonnes
data.columns = data.columns.str.replace("[(/]", "_")
# remplacer les caracteres ')' par '' dans les noms de colonnes
data.columns = data.columns.str.replace("[)]", "")


# In[10]:


# Pie plot types de colonnes
data.dtypes.value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Types de variables')
plt.ylabel('')
fichier ='pieplot_type_variable'+'.png'
plt.savefig(fichier)
plt.show()


# In[11]:


# Pie plot taux de remplissage du jeu de données
pd.DataFrame({'remplissage':[data.notna().mean().mean(),data.isna().mean().mean()]},index=['taux de remplissage','taux de valeurs manquantes']).plot.pie(autopct='%1.1f%%',subplots=True)
plt.title('Taux de remplissage')
plt.ylabel('')
plt.legend(loc='center')
fichier ='pieplot_remplissage'+'.png'
plt.savefig(fichier)
plt.show()


# In[12]:


# Calcul du taux de remplissage  par colonne

plt.figure(figsize=(15, 5))
G = gridspec.GridSpec(1, 1)

ax = plt.subplot(G[0, :])
taux_remplissage = 100-data.isna().mean()*100
ax = taux_remplissage.plot(kind='bar', color='red')
ax.set_title('Taux de remplissage par colonne')
ax.set_xlabel('Colonne')
ax.set_ylabel('Taux de remplissage')
ax.grid(True)
fichier ='taux_remplissage'+'.png'
plt.savefig(fichier)
plt.show()


# In[13]:


data.shape


# # 2. Analyse de fond

# ## Supprimer les lignes vides

# In[14]:


# supprimer les lignes ne contenant que des informations générales 
data.iloc[:,8:].dropna(how='all', axis=0, inplace=True)
data.shape


# ## Supprimer les lignes dupliquées

# In[15]:


# supprimer les doublons
data.drop_duplicates(inplace=True)
print ('Il y a ', data.shape[0], 'lignes et ', data.shape[1],'colonnes dans la base ')


# ## Répartition de la consomation  d'énergie dans la ville 

# In[16]:


import plotly.express as px

fig = px.scatter_mapbox(data, lat="Latitude", lon="Longitude", hover_name="PropertyName", color = "SiteEnergyUse_kBtu"
                       ,color_continuous_scale="viridis", zoom = 10, title= "Répartition de la consomation  en énergie des bâtiments dans la ville de Seattle" )
fig.update_layout(mapbox_style="open-street-map")
#fig.update_layout(title = "Répartition de la consomation  en énergie des bâtiments dans la ville de Seattle" )
fig.update_layout(margin={"r":0,"l":0,"b":0})
fig.show()


# In[17]:


import folium
import folium.plugins

#Coordonnées du centre de Seattle
seattle_lat = 47.6062
seattle_lon = -122.3321

seattle = folium.Map(location=[seattle_lat, seattle_lon], zoom_start=12)

#Clusters
marker_cluster = folium.plugins.MarkerCluster().add_to(seattle)
for lat, lon, in zip(data.Latitude, data.Longitude):
    folium.Marker(location=[lat, lon]).add_to(marker_cluster)

seattle


# ## Sélection et traitement des colonnes utiles

# Une première vérification des variables nous mène à supprimer quelques colonnes inutiles

# In[18]:



colonnes_a_supprimer = ['OSEBuildingID', # identifiant unique 
                        'PropertyName',# n'apporte pas d'information, comme une variable id
                        'City',# seule ville  (Seattle)
                        'State', # même Etat (Washington)
                        'ZipCode', # garder latitude et longitude
                        'YearsENERGYSTARCertified', # information inutile (Années pendant lesquelles la propriété a reçu la certification ENERGY STAR)
                        'TaxParcelIdentificationNumber', #information fiscale inutile
                        'CouncilDistrictCode', # information inutile (code de conseil municipal)
                        'ListOfAllPropertyUseTypes', #plusieurs valeurs qui sont contenues dans les autres colonnes
                        'Outlier', #plusieurs valeurs nulles
                        'Comments' ]# aucune valeur


# **Nouvelle variable 'Building_Age'**

# In[19]:


# A partir de la difference des deux variable "DataYear" et "YearBuilt"
data["Building_Age"] = data["DataYear"] - data["YearBuilt"]
data[["Building_Age", "DataYear", "YearBuilt"]].head()


# In[20]:


# Supprimer les deux variables "DataYear"et "YearBuilt"
for col in ["YearBuilt","DataYear"]:
  colonnes_a_supprimer.append(col)


# **Variable 'NumberofBuildings'**

# In[21]:


def val_colonne(colonne):
  print('nbre_valeurs : ',data[colonne].nunique())
  if data[colonne].dtypes!='object':
    print('Liste_valeurs : ',sorted(data[colonne].unique().tolist()))
  else:
    print('Liste_valeurs : ',data[colonne].unique().tolist())


# In[22]:


val_colonne('NumberofBuildings')


# La valeur 0 est une valeur aberrante, remplacer par 1
# 
# La valeur 111 correspond à une université (google maps) donc c'est une valeur possible

# In[23]:


data.loc[data['NumberofBuildings'] == 0.0,'NumberofBuildings'] = 1 


# **Variable 'NumberofFloors'**

# In[24]:


val_colonne('NumberofFloors')


# La valeur 0 est une valeur aberrante, remplacer par 1

# In[25]:


data.loc[data['NumberofFloors'] == 0.0,'NumberofFloors'] = 1 


# **Variable 'Neighborhood'**

# In[26]:


val_colonne('Neighborhood')


# In[27]:


# minuscule et majuscule, transformer toutes les valeurs en majuscule
data['Neighborhood'] = data['Neighborhood'].apply(lambda x: x.upper().strip())


# **Variables 'PrimaryPropertyType', 'SecondLargestPropertyUseType', 'ThirdLargestPropertyUseType' et 'LargestPropertyUseType'**

# In[28]:


colonne_a_traiter = ['PrimaryPropertyType', 'SecondLargestPropertyUseType', 'ThirdLargestPropertyUseType','LargestPropertyUseType']
for col in colonne_a_traiter:
  val_colonne(col)


# Plusieurs valeurs.
# Créer un dictionnaire de données pour regrouper les valeurs.

# In[29]:


typeDict = {
    #Education
    'Adult Education' : 'Education',
    'College/University' : 'Education',
    'K-12 School' : 'Education',
    'Other - Education' : 'Education',
    'Pre-school/Daycare' : 'Education',
    'SPS-District K-12' : 'Education',
    'University' : 'Education',
    'Vocational School' : 'Education',

    #Entertainment
    'Bar/Nightclub' : 'Entertainment',
    'Convention Center' : 'Entertainment',
    'Fitness Center/Health Club/Gym' : 'Entertainment',
    'Movie Theater' : 'Entertainment',
    'Museum' : 'Entertainment',
    'Other - Entertainment/Public Assembly' : 'Entertainment',
    'Other - Recreation' : 'Entertainment',
    'Performing Arts' : 'Entertainment',
    'Social/Meeting Hall' : 'Entertainment',
    'Swimming Pool' : 'Entertainment',
    
    #Food
    'Fast Food Restaurant' : 'Food',
    'Food Sales' : 'Food',
    'Food Service' : 'Food',
    'Other - Restaurant/Bar' : 'Food',
    'Restaurant' : 'Food',
    'Restaurant\n' : 'Food',

    #Healthcare
    'Hospital' : 'Healthcare',
    'Hospital (General Medical & Surgical)' : 'Healthcare',
    'Laboratory' : 'Healthcare',
    'Medical Office' : 'Healthcare',
    'Other/Specialty Hospital' : 'Healthcare',
    'Outpatient Rehabilitation/Physical Therapy' : 'Healthcare',
    'Residential Care Facility' : 'Healthcare',
    'Urgent Care/Clinic/Other Outpatient' : 'Healthcare',

    #Lodging
    'Hotel' : 'Lodging',
    'High-Rise Multifamily' : 'Lodging',
    'Low-Rise Multifamily' : 'Lodging',
    'Mid-Rise Multifamily' : 'Lodging',
    'Multifamily Housing' : 'Lodging',
    'Other - Lodging/Residential' : 'Lodging',
    'Prison/Incarceration' : 'Lodging',
    'Residence Hall' : 'Lodging',
    'Residence Hall/Dormitory' : 'Lodging',
    'Senior Care Community' : 'Lodging',
        
    #Manufacturing/Warehouse
    'Manufacturing/Industrial Plant' : 'Industrial',
    'Other - Utility' : 'Industrial',
    'Distribution Center' : 'Industrial',
    'Distribution Center\n' : 'Industrial',
    'Non-Refrigerated Warehouse' : 'Industrial',
    'Refrigerated Warehouse' : 'Industrial',
    'Self-Storage Facility' : 'Industrial',
    'Self-Storage Facility\n' : 'Industrial',
    'Warehouse' : 'Industrial',

    #Office
    'Large Office' : 'Office',
    'Office' : 'Office',
    'Small- and Mid-Sized Office' : 'Office',
    'Other - Services' : 'Office',
    'Data Center' : 'Office', #Tech
    'Other - Technology/Science' : 'Office', #Tech
    'Bank Branch' : 'Office', #Banking
    'Financial Office' : 'Office', #Banking
    
    #Public Services
    'Courthouse' : 'PublicService',
    'Fire Station' : 'PublicService',
    'Library' : 'PublicService',
    'Police Station' : 'PublicService',
    'Other - Public Services' : 'PublicService',
    'Worship Facility' : 'PublicService',
    
    #Retail
    'Automobile Dealership' : 'Retail',
    'Convenience Store without Gas Station' : 'Retail',
    'Enclosed Mall' : 'Retail',
    'Lifestyle Center' : 'Retail',
    'Other - Mall' : 'Retail',
    'Personal Services (Health/Beauty, Dry Cleaning, etc)' : 'Retail',
    'Repair Services (Vehicle, Shoe, Locksmith, etc)' : 'Retail',
    'Retail Store' : 'Retail',
    'Strip Mall' : 'Retail',
    'Supermarket / Grocery Store' : 'Retail',
    'Supermarket/Grocery Store' : 'Retail',
    'Wholesale Club/Supercenter' : 'Retail',
    
    #Other
    'Mixed Use Property' : 'Other',
    'Other' : 'Other',
    'Parking' : 'Other',
}


# In[30]:




for col in colonne_a_traiter: 
    data[col] = data[col].replace(typeDict, regex=False)


# **Variables'ComplianceStatus'**

# In[31]:


val_colonne('ComplianceStatus')


# In[32]:


data['ComplianceStatus'].value_counts(normalize=True)*100


# **Variables 'DefaultData'**

# In[33]:


val_colonne('DefaultData')


# In[34]:


data['DefaultData'].value_counts(normalize=True)*100


# **Variable 'Address'**
# 
# A premier vu, cette variable me parait inutile puisque j'ai gardé les variables latitude et longitude. Mais la phrase "Fais bien attention au traitement des différentes variables, à la fois pour trouver de nouvelles informations (peut-on déduire des choses intéressantes d’une simple **adresse** ?)" dans l'énoncé du projet m'informe qu'il faut analyser cette variable.
# 

# In[35]:


val_colonne('Address')


# In[36]:


# chercher si l'adresse contient les mots "avenue", "rue", etc
def contenu_adress(adresse):
    
    if (re.search('ST|STREET', adresse,re.IGNORECASE)):
        return 'STREET'
    elif (re.search('AVE|AVENUE', adresse,re.IGNORECASE)):
        return 'AVENUE'

    elif (re.search('WAY | ROUTE| Road', adresse,re.IGNORECASE)):
        return 'WAY'
    else:
        return 'OTHER'


# In[37]:


data['New_Address'] = data['Address'].apply(contenu_adress)


# In[38]:


# Vérifier la consommation d'énergie et l'emission de gaz /contenu adresse
plt.figure(figsize=[15, 10])

sns.pairplot(data[['Electricity_kBtu','SiteEnergyUseWN_kBtu','NaturalGas_kBtu','TotalGHGEmissions','New_Address']], kind='scatter', hue='New_Address')

plt.show()


# D'après cette figure, la consommation d'énergie et l'emission de gaz dépondent du contenu de l'adrese. On garde la nouvelle variable 'New_Address' et on supprime l'ancienne variable 'Address'

# In[39]:


colonnes_a_supprimer.append('Address')


# **Variables énergitiques**
# 
# ENERGYSTARScore : An EPA calculated 1-100 rating that assesses a property’s overall energy performance, based on national data to control for differences among climate, building uses, and operations. A score of 50 represents the national median.
# 
# SiteEUI(kBtu/sf):	Site Energy Use Intensity (EUI) is a property's Site Energy Use divided by its gross floor area. Site Energy Use is the annual amount of all the energy consumed by the property on-site, as reported on utility bills. Site EUI is measured in thousands of British thermal units (kBtu) per square foot.
# 
# SiteEUIWN(kBtu/sf) :	Weather Normalized (WN) Site Energy Use Intensity (EUI) is a property's WN Site Energy divided by its gross floor area (in square feet). WN Site Energy is the Site Energy Use the property would have consumed during 30-year average weather conditions. WN Site EUI is measured in measured in thousands of British thermal units (kBtu) per square foot.
# 
# SourceEUI(kBtu/sf) :	Source Energy Use Intensity (EUI) is a property's Source Energy Use divided by its gross floor area. Source Energy Use is the annual energy used to operate the property, including losses from generation, transmission, & distribution. Source EUI is measured in thousands of British thermal units (kBtu) per square foot.
# 
# SourceEUIWN(kBtu/sf) : Weather Normalized (WN) Source Energy Use Intensity (EUI) is a property's WN Source Energy divided by its gross floor area. WN Source Energy is the Source Energy Use the property would have consumed during 30-year average weather conditions. WN Source EUI is measured in measured in thousands of British thermal units (kBtu) per square foot.
# 
# SiteEnergyUse(kBtu) :	The annual amount of energy consumed by the property from all sources of energy.
# 
# SiteEnergyUseWN(kBtu) :	The annual amount of energy consumed by the property from all sources of energy, adjusted to what the property would have consumed during 30-year average weather conditions.
# 
# SteamUse(kBtu) :	The annual amount of district steam consumed by the property on-site, measured in thousands of British thermal units (kBtu).
# 
# Electricity(kWh) :	The annual amount of electricity consumed by the property on-site, including electricity purchased from the grid and generated by onsite renewable systems, measured in kWh.
# 
# Electricity(kBtu) :	The annual amount of electricity consumed by the property on-site, including electricity purchased from the grid and generated by onsite renewable systems, measured in thousands of British thermal units (kBtu).
# 
# NaturalGas(therms) :	The annual amount of utility-supplied natural gas consumed by the property, measured in therms.
# 
# NaturalGas(kBtu) :	The annual amount of utility-supplied natural gas consumed by the property, measured in thousands of British thermal units (kBtu).

# *Unités énergitique*
# 
# kWh : kilowattheure
# 
# Therm : unité anglo-saxonne d'énergie égale à 100 000 British thermal unit (BTU)
# 
# kBtu : British thermal unit (=0,293071 KWh)
# 
# Je vais garder l'unité kBtu

# In[40]:


for col in ['Electricity_kWh', 'NaturalGas_therms','Electricity_kBtu','NaturalGas_kBtu']:
  colonnes_a_supprimer.append(col)


# *Suffixe WN* 
# 
# représente la consommation d'énergie que la propriété aurait consommée dans des conditions météorologiques moyennes sur 30 ans. 
# On verra dans la partie features importance si on les gardes ou on les supprimes

# *Suffixe _sf* 
# 
# représente la consommation d'énergie divisée par la surface. 
# Ce sont des variables calculées à partir d'autres donc  à supprimer

# In[41]:


for col in ['SiteEUIWN_kBtu_sf','SourceEUIWN_kBtu_sf','SourceEUI_kBtu_sf','PropertyGFABuilding_s',  'SiteEnergyUseWN_kBtu']:
  colonnes_a_supprimer.append(col)


# ## supprimer les bâtiments habitables

# In[42]:


data.shape


# In[43]:


val_colonne('BuildingType')


# In[44]:


data.drop(data.loc[data['BuildingType']=='Multifamily MR (5-9)'].index, inplace=True)
data.drop(data.loc[data['BuildingType']=='Multifamily LR (1-4)'].index, inplace=True)
data.drop(data.loc[data['BuildingType']=='Multifamily HR (10+)'].index, inplace=True)


# In[45]:


data.shape


# ## Supprimer les colonnes sélectionnées

# In[46]:


data = data.drop(colonnes_a_supprimer, 1)


# In[47]:


data.shape


# # 3.  Imputation des valeurs manquantes

# In[48]:


def informations_valeurs_manqantes(df):
  print('Nombre de valeurs manquantes par colonne')
  msno.bar(df)
  print(100*'*')
  print('Matrice de chaleur des valeurs manquantes')
  msno.heatmap(df)
  print(100*'*')
  print('Dendogramme des valeurs manquantes')
  msno.dendrogram(df)
  print(100*'*')


# In[49]:


informations_valeurs_manqantes(data)


# **Variable ENERGYSTARScore**
# 
# On doit étudier la pertinence de cette variable  donc une imputation biaiserai notre étude. Pour le moment, je vais laisser cette variable telle quelle est.

# **Variables cibles**
# 
# Supprimer les valeurs nulles des deux variables cibles SiteEnergyUse_kBtu  et
# TotalGHGEmissions.

# In[50]:


data = data[~(data['SiteEnergyUse_kBtu'].isna() | data['TotalGHGEmissions'].isna())]


# **Variables categorielles**

# In[51]:


# Le nombre de valeurs de chaque variable objet
data.select_dtypes(['object']).nunique()


# In[52]:


data.select_dtypes(['object']).isnull().mean()


# Si PrimaryPropertyType   manquantes, on impute par la valeur "unknown"          
# 
# Si SecondLargestPropertyUseType manquantes, on impute par la variable  PrimaryPropertyType
# 
# Si ThirdLargestPropertyUseType, on impute par la variable SecondLargestPropertyUseType
# 
# Si LargestPropertyUseType   manquantes, on impute par la variable  PrimaryPropertyType

# In[53]:


data['PrimaryPropertyType'] = data['PrimaryPropertyType'].fillna("unknown")
data['SecondLargestPropertyUseType'] = data['SecondLargestPropertyUseType'].fillna(data['PrimaryPropertyType'])
data['ThirdLargestPropertyUseType'] = data['ThirdLargestPropertyUseType'].fillna(data['SecondLargestPropertyUseType'])
data['LargestPropertyUseType'] = data['LargestPropertyUseType'].fillna(data['PrimaryPropertyType'])


# In[54]:


# verifier s'il y a encore des données manquantes dans les variables categorielles
data.select_dtypes(['object']).isnull().mean()


# **Variables continues**

# In[55]:


def imputation_mediane(df,col_imputation,col_groupby):
  df[col_imputation].fillna(df.groupby(col_groupby)[col_imputation].transform('median'), inplace = True)
  # Il reste quelques valeurs nulles dont le groupement n'a pa de mediane. On les remplace par 0
  df[col_imputation].fillna(0, inplace = True)


# In[56]:


def distribution_avant_apres(df_avant, df_apres, colonne_cible):
  colonne_cible_apres_imputation = colonne_cible+'_apres_imputation'
  sns.kdeplot(colonne_cible, data = df_avant, label=colonne_cible)
  sns.kdeplot(colonne_cible, data = df_apres, label=colonne_cible_apres_imputation)
  plt.title(f'comparaison des distributions prediction colonne {colonne_cible}')
  plt.legend()
  plt.show()


# In[57]:



colonnes_median = {'SecondLargestPropertyUseTypeGFA':'SecondLargestPropertyUseType',
                  'ThirdLargestPropertyUseTypeGFA':'ThirdLargestPropertyUseType',
                  'LargestPropertyUseTypeGFA':'LargestPropertyUseType'}
# garder une copie des données avant imputation
data_median = data[list(colonnes_median.keys())].copy()
# imputation par la mediane des categories (groupby sur les colonne 'main_category_fr')
for col_imputation,col_groupby in colonnes_median.items():
 imputation_mediane(data,col_imputation,col_groupby)
 
# 2eme copie après imputation pour faire la comparaison des distributins
data_median_imputed = data[list(colonnes_median.keys())].copy()


# In[58]:


for colonne_cible in list(colonnes_median.keys()):
  distribution_avant_apres(data_median, data_median_imputed, colonne_cible)


# In[59]:


for colonne in list(colonnes_median.keys()):
    data[colonne] = data_median_imputed[colonne]


# In[60]:


# verifier s'il y a encore des données manquantes autre que la variable ENERGYSTARScore  
data.isnull().mean()


# In[61]:


data.shape


# # 4.  Analyse valeur aberrante

# **Boite à moustaches**

# In[62]:


for colonne in data.select_dtypes(include=[np.number]).columns.tolist():
    sns.boxplot(x=colonne, data = data,whis=[5, 95])
    plt.title(('Boite a moustache de la colonne ' + colonne))
    #fichier ='moustache_'+colonne+'.png'
    #plt.savefig(fichier)
    plt.show()


# D'après ces boites à moustaches, nous observons qu'il ya des valeurs aberrantes pour la majorité des colonnes.
# Je vais traiter ces valeurs aberranes par la méthode des quartiles (0.05 comme quartile inf et 0.95 comme quartile sup pour ne pas perdre beaucoups d'observations). Le nombre d'observations est déjà petit (environ 1600).
# 

# In[63]:


colonnes_aberrantes = ['PropertyGFATotal',
                      'PropertyGFAParking',
                      'LargestPropertyUseTypeGFA',
                      'SecondLargestPropertyUseTypeGFA',
                      'ThirdLargestPropertyUseTypeGFA',
                      'SiteEUI_kBtu_sf',
                      'SteamUse_kBtu',
                      'TotalGHGEmissions',
                      'GHGEmissionsIntensity',]


# In[64]:


def traitement_valeurs_aberrantes(data,colonne):
    Q1 = data[colonne].quantile(0.05)
    Q3 = data[colonne].quantile(0.95)
    borneInf = Q1 - 1.5*(Q3 - Q1)
    borneSup = Q3 + 1.5*(Q3 - Q1)    
    data.drop(data.loc[data[colonne] > borneSup].index, inplace = True)
    data.drop(data.loc[data[colonne] < borneInf].index, inplace = True)


# In[65]:



for colonne in colonnes_aberrantes:
  traitement_valeurs_aberrantes(data,colonne)


# In[66]:


# Supprimer les valeurs négatives

data = data.loc[data['SiteEnergyUse_kBtu'] > 0]
data = data.loc[data['TotalGHGEmissions'] > 0]


# In[67]:


data.shape


# **ré-indexer le dataframe après tous les traitements effectués**

# In[68]:


data=data.reset_index(drop = True)


# # 5.  Analyse univariée

# In[69]:


def analyse_univariee(data,colonne,label):
    print(f'moyenne : {round(data[colonne].mean(),2)}')
    print(f'mediane : {round(data[colonne].median(),2)}')
    print(f'mode : {round(data[colonne].mode(),2)}')
    print(f'variance : {round(data[colonne].var(),2)}')
    print(f'skewness : {round(data[colonne].skew(),2)}')
    print(f'kurtosis : {round(data[colonne].kurtosis(),2)}')
    print(f'ecart type : {round(data[colonne].std(),2)}')
    print(f'min : {round(data[colonne].min(),2)}')
    print(f'25% : {round(data[colonne].quantile(0.25),2)}')
    print(f'50% : {round(data[colonne].quantile(0.5),2)}')
    print(f'75% : {round(data[colonne].quantile(0.75),2)}')
    print(f'max : {round(data[colonne].max(),2)}')
    print(colored('Interprétation', 'red', attrs=['bold']))
    if np.floor(data[colonne].skew())==0:
        print('la distribution de la colonne '+colonne +' est symétrique')
    elif round(data[colonne].skew(),2)>0:
        print('la distribution de la colonne '+colonne + ' est étalée à droite')
    else:
        print('la distribution de la colonne '+colonne +' est étalée à gauche')
    
    if np.floor(data[colonne].kurtosis())==0:
        print('la distribution de la colonne '+colonne +' a le même aplatissement que la distribution normale')
    elif round(data[colonne].kurtosis(),2)>0:
        print('la distribution de la colonne '+colonne + ' est moins aplatie que la distribution normale')
    else:
        print('la distribution de la colonne '+colonne +' est plus aplatie que la distribution normale')
                   
    plt.figure(figsize=(15, 5))
    plt.subplot( 1,2 ,1)
    sns.boxplot(data[colonne], width=0.5, color='red')
    plt.title('Boite a moustache de la colonne '+label,fontsize=15)
    plt.subplot(1,2,2 )
    sns.histplot(data[colonne], kde=True, color='blue')
    plt.title('histogramme de la colonne  '+label,fontsize=15)        
    plt.show()       
    plt.tight_layout()
                  


# In[70]:


colonnes_numeriques = data.select_dtypes(include=[np.number]).columns.tolist()


# In[71]:


for colonne in colonnes_numeriques:
    print(colored(150*'*', 'blue', attrs=['bold']))
    print(colored('Analyse de la colonne '+colonne, 'red', attrs=['bold']))
    analyse_univariee(data,colonne,str(colonne))
    #fichier ='univarie_'+colonne+'.png'
    #plt.savefig(fichier)
print(colored(150*'*', 'blue', attrs=['bold']))


# # 6.  Analyse multivariée

# In[72]:


def matrice_correlation(data,colonnes_a_analyser):
  plt.rcParams["figure.figsize"]=[15,7]
  data = data[colonnes_a_analyser]
  mask = np.triu(np.ones_like(data.corr(), dtype=bool))
  sns.heatmap(data.corr(), vmin=-1, vmax=1,annot=True,fmt = ".2f",mask=mask)
  plt.title('matrice de corrélation entre les colonnes ')
  fichier ='heatmap_fr'+'.png'
  plt.savefig(fichier)
  plt.show()


# In[73]:


colonnes_numeriques= data.select_dtypes(include=[np.number]).columns.tolist()
matrice_correlation(data,colonnes_numeriques)


# D'après la matrice de corrélation, nous remarquons :
# 
# 
# Forte corrélation entre  LargestPropertyUseTypeGFA et PropertyGFATotal
# 
# Forte corrélation entre  SecondLargestPropertyUseTypeGFA et  PropertyGFATotal
# 
# Forte corrélation entre  SiteEnergyUse_kBtu et PropertyGFATotal
# 
# Forte corrélation entre SiteEnergyUse_kBtu et LargestPropertyUseTypeGFA
# 
# Forte corrélation entre SiteEnergyUse_kBtu et TotalGHGEmissions
# 
# Forte corrélation entre SiteEUI_kBtu_sf et GHGEmissionsIntensity
# 
# 
# Il faut supprimer l'une de ces variables ou en créer des nouvelles 

# In[74]:


# Vérification d'un exemple de corrélation à  l'aide d'un shéma
sns.regplot(data=data, x='LargestPropertyUseTypeGFA', y='PropertyGFATotal')
plt.show()


# # 7.  Feature engineering

# ## Transformation des variables

# In[75]:


# Déviser quelques colonnes par la surface
def poucentage_surface(data,new_colonne,old_colonne,colonne_surface):
  data[new_colonne] = data[old_colonne]/data[colonne_surface]
  data = data.drop(old_colonne,1)


# In[76]:


colonnes_pourcentage = {'Rate_Parking':'PropertyGFAParking',
                        'Rate_LargestPropertyUseType':'LargestPropertyUseTypeGFA',
                        'Rate_SecondLargestPropertyUseType':'SecondLargestPropertyUseTypeGFA',
                        'Rate_ThirdLargestPropertyUseType':'ThirdLargestPropertyUseTypeGFA'}

colonne_surface =  'PropertyGFATotal'  

for new_colonne,old_colonne in colonnes_pourcentage.items():
  print(new_colonne,'<<-------',old_colonne)
  poucentage_surface(data,new_colonne,old_colonne,colonne_surface)


# In[ ]:


# Vérifier la corrélation après cette transformation
dataaa = data.copy()
dataaa = dataaa.drop(['PropertyGFAParking','LargestPropertyUseTypeGFA','SecondLargestPropertyUseTypeGFA','ThirdLargestPropertyUseTypeGFA'], 1)
colonnes_numeriques= dataaa.select_dtypes(include=[np.number]).columns.tolist()
matrice_correlation(dataaa,colonnes_numeriques)


# Nous remarquons après cette transformation que la majorité des corrélations ont été supprimées. Il reste 3 corrélations. On va les revoir dans la partie features importance

#  ## One-hot encoding
# 
# Le one hot encoding est la méthode la plus populaire pour transformer une variable catégorique en variable numérique. Sa popularité réside principalement dans la facilité d’application. De plus, pour beaucoup de problèmes, elle donne de bons résultats. Son principe est le suivant :
# 
# Considérons une variable catégorique X qui admet K modalités m1, m2, …, mK. Le one hot encoding consiste à créer K variables indicatrices, soit un vecteur de taille K qui a des 0 partout et un 1 à la position i correspondant à la modalité mi. On remplace donc la variable catégorique par K variables numériques.

# In[ ]:


def encodage_categorielle(data,colonne):
  df_encodage=pd.get_dummies(data[colonne],prefix=colonne)
  return (df_encodage)


# In[ ]:


colonnes_categoriques = data.select_dtypes('object').columns
data_encodage_categorielle = pd.DataFrame(index=data.index)
for colonne in colonnes_categoriques:
  df_encodage = encodage_categorielle(data,colonne)
  data_encodage_categorielle = pd.concat([data_encodage_categorielle,df_encodage],axis=1)


# In[ ]:


data.shape


# In[ ]:


data = data.drop(colonnes_categoriques,1)


# In[ ]:


data = pd.concat([data,data_encodage_categorielle],axis=1)


# In[ ]:


# remplacer les caracteres '(','/' par '_' dans les noms de colonnes
data.columns = data.columns.str.replace("[ /-]", "_")


# In[ ]:


data.shape


# # 8.  Features Importance
# 

# **Target 1 SiteEnergyUse_kBtu**

# In[ ]:


data = data.astype(float)


# In[ ]:


dataa=data.copy()


# In[ ]:


# Supprimer la variable EnergystarScore parce qu'elle contient plusieurs valeurs manquantes et la variable target 2
data = data.drop(['ENERGYSTARScore','TotalGHGEmissions'],1)


# In[ ]:


data = data.drop(['SiteEUI_kBtu_sf', 'SteamUse_kBtu',  'GHGEmissionsIntensity'],1)


# In[ ]:


def features_importance(data,target):
  X = data.copy()
  X = X.drop(target, 1)
  y = data[target]  
  plt.figure(figsize=[10, 10])
  rf_model = RandomForestRegressor( n_jobs=-1,random_state=0,n_estimators=10)
  viz = FeatureImportances(rf_model, relative=True,topn=20)
  #viz.fit(X.iloc[:,0:15], y)
  viz.fit(X, y)


# In[ ]:


features_importance(data,'SiteEnergyUse_kBtu')


# Il s'avère que les colonnes  'LargestPropertyUseTypeGFA' et 'PropertyGFATotal'  sont  importants pour notre modèle de prédiction du target 1. On peut les garder 

# In[ ]:


data1 = dataa.copy()
colonnes_numeriques1 =colonnes_numeriques.copy()


# In[ ]:


data1 = data1.drop(['SiteEUI_kBtu_sf', 'SteamUse_kBtu', 'TotalGHGEmissions'],1)


# In[ ]:


for elem in ['SiteEUI_kBtu_sf', 'SteamUse_kBtu', 'TotalGHGEmissions']:
    colonnes_numeriques1.remove(elem)


# In[ ]:


# Vérifier les corrélations après la suppressions de quelques variables
matrice_correlation(data1,colonnes_numeriques1)


# **Target 2 TotalGHGEmissions**

# In[ ]:


data = dataa.copy()


# In[ ]:


# Supprimer la variable EnergystarScore parce qu'elle contient plusieurs valeurs manquantes et la variable target 1
data = data.drop(['ENERGYSTARScore','SiteEnergyUse_kBtu','GHGEmissionsIntensity'],1)


# In[ ]:


features_importance(data,'TotalGHGEmissions')


# Il s'avère que les colonnes   'LargestPropertyUseTypeGFA' et 'PropertyGFATotal'  sont  importantes pour notre modèle de prédiction du target 2. On peut les garder. 
# On peut supprimer la colonne 'GHGEmissionsIntensity', puisque elle représente des valeurs qu'on veut prédire. 
# 

# In[ ]:


data2 = dataa.copy()
colonnes_numeriques2 =colonnes_numeriques.copy()


# In[ ]:


data2 = data2.drop(['GHGEmissionsIntensity','SiteEnergyUse_kBtu'],1)


# In[ ]:


for elem in ['GHGEmissionsIntensity','SiteEnergyUse_kBtu']:
    colonnes_numeriques2.remove(elem)


# In[ ]:


# Vérifier les corrélations après la suppressions de quelques variables
matrice_correlation(data2,colonnes_numeriques2)


# # 9.  Sauvegarde du jeu de données après nettoyage

# In[ ]:


data1.shape


# In[ ]:


data2.shape


# In[ ]:


data1.to_csv('data_energy_nettoye.csv', encoding='utf_8',index=False)


# In[ ]:


data2.to_csv('data_gaz_nettoye.csv', encoding='utf_8',index=False)

