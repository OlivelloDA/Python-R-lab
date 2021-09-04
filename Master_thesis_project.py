import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import pylab 
import tensorflow as tf
from scipy import stats
import seaborn as sns;sns.set()
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Input
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn import preprocessing
from tensorflow.python.keras import backend as K
import random
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_gamma_deviance



np.random.seed(697)
alt.renderers.enable('altair_viewer')
alt.data_transformers.disable_max_rows()
df = pd.read_csv(r'Z:\RW_FOLDER\projects\cca_data\claudio_thesis\thesis_claudio_database.csv')
df['claim_declar_dt'] = pd.to_datetime(df['claim_declar_dt'])
df['claim_dt'] = pd.to_datetime(df['claim_dt'])
df_sample = df
df_sample= df_sample[df_sample['valsin']<30000]
df_sample =df_sample[df_sample['pieces']>0]
df_sample =df_sample[df_sample['valsin']>0]
df_sample = df_sample[0:200000]
df_sample = df_sample.dropna(thresh=82) #drop those rows that have more than 7 nan = 6% of the cars
''' 1. DATA CLEANING AND EXPLORATION'''

'''
#missing and nan values analysis
df['antivo'].isna().sum()/len(df)*100 #4.8%
(df_sample['antivo'].isna().sum())#1531 nan , 15k Y
df['claim_postal_cd'].isna().sum()/len(df)*100 #48.4%
df['claim_country_cd'].isna().sum()/len(df)*100 #24.9%
df['claim_time'].isna().sum()/len(df)*100 #17.90%
value = df['claim_time'].value_counts()
'''

'''-------------------CLEANING--------------------------
1) I drop the entries
2) I substitute it with the median/mean
3) I substitute nan with the mode distribution
4) I drop the variable
'''
#delta time = date declaration of the claim - date the claim happened
df_sample['deltaTime'] = df_sample['claim_declar_dt']-df_sample['claim_dt']
df_sample['deltaTime']= df_sample['deltaTime'].dt.days
df_sample['deltaTime'] = df_sample['deltaTime'][((df_sample['deltaTime']>=0) & (df_sample['deltaTime']<100))]
#df_sample['deltaTime'].value_counts()
df['circumstance_cd'].isna().sum()/len(df)*100 #13.4%
df_sample['circumstance_cd'] = df_sample['circumstance_cd'].fillna(df_sample['circumstance_cd'].median())
df_sample['claim_time'].isna().value_counts()#25%
df_sample = df_sample.dropna(subset=['deltaTime','claim_responsibility_rate','claim_time'])#this deletes 30% of the variables!
df_sample[df_sample['claim_time']==0].value_counts()
df_sample = df_sample[df_sample['claim_time']!=0]# I delete zero values because I don't know when they are missing or midnight. However the real midnight drops should be around 100
df_sample['claim_time'] = df_sample['claim_time'][df_sample['claim_time'].isna()==False].astype(str).str[0:2]
df_sample['claim_time'] = df_sample['claim_time'].fillna(11)#11
df_sample['claim_time'] = df_sample[df_sample['claim_time']!= '0.']
#drop the rows where the variable takes the value nan
df_sample['bodyshop_postal_code'].isna().value_counts()#23% --> DROP
df_sample['claim_postal_cd'].isna().value_counts()#50% --> DROP
df_sample['claim_country_cd'].isna().value_counts()#23%--> DROP

df_sample['valass'].isna().value_counts()#0%
df_sample[df_sample['valass']==0]#25%
df_sample['valass'].loc[df_sample['valass'] == 0] = df_sample['valass'].median()
#df_sample.loc[df_sample['valass']] = [df_sample['valsin'] if x > df_sample['valass'] else df_sample['valass'] for x in df_sample['valsin']]
df_sample['pieces'].isna().value_counts()#0%
df_sample[df_sample['pieces']==0]#8%
df_sample['pieces'].loc[df_sample['pieces'] == 0] =df_sample['pieces'].median()
 
df_sample['PER_MOS_Social_Class_Code'].isna().value_counts()#32%
df_sample[df_sample['PER_MOS_Social_Class_Code']==0]#0%
#replace the missing with mode distribution. I believe it makes sense just for some variables, I insert bias, tradeoff between bias and advantages of having the variable in the model
counts = df_sample['PER_MOS_Social_Class_Code'].value_counts()
dist = stats.rv_discrete(values=(np.arange(counts.shape[0]), 
                                 counts/counts.sum()))
fill_idxs = dist.rvs(size=df_sample.shape[0] - df_sample['PER_MOS_Social_Class_Code'].count())
df_sample['PER_MOS_Social_Class_Code'].loc[df_sample['PER_MOS_Social_Class_Code'].isna()] = pd.Series(counts.iloc[fill_idxs].index)
df_sample = df_sample.dropna(subset=['PER_MOS_Social_Class_Code'])

df_sample['PER_MOS_Urbanization_Lvl_cd'].isna().value_counts()#28% HOWEVER, with the dropping in the last variable, it falls to 0%
df_sample[df_sample['PER_MOS_Urbanization_Lvl_cd']==0]#0%
counts = df_sample['PER_MOS_Urbanization_Lvl_cd'].value_counts()
dist = stats.rv_discrete(values=(np.arange(counts.shape[0]), 
                                 counts/counts.sum()))
fill_idxs = dist.rvs(size=df_sample.shape[0] - df_sample['PER_MOS_Urbanization_Lvl_cd'].count())
df_sample['PER_MOS_Urbanization_Lvl_cd'].loc[df_sample['PER_MOS_Urbanization_Lvl_cd'].isna()] = pd.Series(counts.iloc[fill_idxs].index)

df_sample['PER_MOS_Net_Income_Average'].isna().value_counts()#30% HOWEVER, with the dropping in the last variable, it falls to 0%
df_sample[df_sample['PER_MOS_Net_Income_Average']==0]#0.1%
counts = df_sample['PER_MOS_Net_Income_Average'].value_counts()
dist = stats.rv_discrete(values=(np.arange(counts.shape[0]), 
                                 counts/counts.sum()))
fill_idxs = dist.rvs(size=df_sample.shape[0] - df_sample['PER_MOS_Net_Income_Average'].count())
df_sample['PER_MOS_Net_Income_Average'].loc[df_sample['PER_MOS_Net_Income_Average'].isna()] = pd.Series(counts.iloc[fill_idxs].index)

df_sample[df_sample['PER_gender']==0]#0%
counts = df_sample['PER_gender'].value_counts()
dist = stats.rv_discrete(values=(np.arange(counts.shape[0]), 
                                 counts/counts.sum()))
fill_idxs = dist.rvs(size=df_sample.shape[0] - df_sample['PER_gender'].count())
df_sample['PER_gender'].loc[df_sample['PER_gender'].isna()] = pd.Series(counts.iloc[fill_idxs].index)


df_sample.drop(df_sample['PER_gender'].loc[df_sample['PER_gender'].isna()].index, inplace = True)
df_sample.drop(df_sample['PER_occupation'].loc[df_sample['PER_occupation'].isna()].index, inplace = True)
df_sample.drop(df_sample['PER_postal_cd'].loc[df_sample['PER_postal_cd'].isna()].index, inplace = True)


df_sample['private_corporate_cd_C'].isna().value_counts()#0.1%
df_sample[df_sample['private_corporate_cd_C']==0]#0%
df_sample['private_corporate_cd_C'] = df_sample['private_corporate_cd_C'].fillna(df_sample['private_corporate_cd_C'].value_counts().index[0])

df_sample['role_cd'].isna().value_counts()#0.1%
df_sample[df_sample['role_cd']==0]#0%
df_sample['role_cd'] = df_sample['role_cd'].fillna(df_sample['role_cd'].value_counts().index[0])

df_sample['PER_langage_cd_C'].isna().value_counts()#0.1%
df_sample[df_sample['PER_langage_cd_C']==0]#0%
df_sample['PER_langage_cd_C'] = df_sample['PER_langage_cd_C'].fillna(df_sample['PER_langage_cd_C'].value_counts().index[0])

df_sample['kiloml'].isna().value_counts()#0%
df_sample[df_sample['kiloml']==0]#1%
df_sample = df_sample[df_sample['kiloml']!=0] 

df_sample['antivo'].isna().value_counts()#0.1%
df_sample[df_sample['antivo']==0]#0%
df_sample['antivo'] = df_sample['antivo'].fillna(df_sample['antivo'].value_counts().index[0])

df_sample['bodyshop_postal_code'].isna().value_counts()#0.1%
df_sample[df_sample['bodyshop_postal_code']==0]#0%
df_sample['bodyshop_postal_code'] = df_sample['bodyshop_postal_code'].fillna(df_sample['bodyshop_postal_code'].value_counts().index[0])

df_sample['PER_driving_licence_cat_cd'].isna().value_counts()#50% --> DROP
df_sample['PER_nationality_cd'].value_counts()#belgium is too predominant --> DROP
df_sample['WDM_neighbourhood_id'].isna().value_counts()#75% --> DROP
df_sample['WDM_street_id'].isna().value_counts()#75% --> DROP


df_sample['private_corporate_cd_C'].isna().value_counts()#0.1%
df_sample[df_sample['private_corporate_cd_C']==0]#0%
df_sample['private_corporate_cd_C'] = df_sample['private_corporate_cd_C'].fillna(df_sample['private_corporate_cd_C'].value_counts().index[0])

#I drop region1,region2,region3 because it give me the same information as region
#I drop smart_expertise_period because all the missions have the same value --> no information
df_sample = df_sample.drop(columns = ['claim_dt','claim_declar_dt','claim_opening_dt','lregion','lprovince','claim_postal_cd'
                          ,'claim_no','dcloex','claim_country_cd','nummis',
                          'voi','cam','mot','PER_driving_licence_cat_cd',
                          'PER_nationality_cd','WDM_street_id','WDM_neighbourhood_id','region1','region2','region3','smart_expertise_period'])

#normalisation
columns_to_scale = ['valass','valsin','kiloml','kilowa','pieces','duration_mission_sent_closed',
                    'claim_time']
df_sample[columns_to_scale]=df_sample[columns_to_scale].apply(lambda x: (x-x.min())/(x.max()-x.min()))

categorical_columns = ['coll39', 'manexp','carbur', 'manexp','effrac', 'antivo','degatx',
                       'degat0','degat1','degat2','degat3','degat4','degat5','degat6','degat7','degat8','degat9',
                       'degat_tot', 'vehimmo', 'contecA','contecB', 'contecR','contecN', 'contec1','contec2','contec3',
                       'contec4', 'region', 'province','bodyshop_postal_code', 'camion',
                       'autoclosure','gamme1', 'gamme2', 'gamme3', 'gamme4', 'gamme41','gamme5','gamme6','cat1',
                       'cat2', 'cat3', 'cat4', 'cat5','cat6', 'cat7','marque', 'modell', 'claim_branch_cd', 
                       'circumstance_cd','claim_responsibility_rate','PER_MOS_Social_Class_Code','PER_MOS_Urbanization_Lvl_cd',
                      'PER_MOS_Net_Income_Average', 'private_corporate_cd_C','role_cd','PER_langage_cd_C', 
                      'PER_gender','PER_postal_cd','PER_occupation']

df_checkpoint1 = df_sample
#df_sample = df_checkpoint1



'''--------------------PLOTS-------------------------'''

df_sample_plot = df_sample[['claim_time','valsin','deltaTime']]
chart6 = alt.Chart(df_sample_plot).mark_bar().encode(alt.X('claim_time'), alt.Y('mean(valsin)'))
chart6.show()
chart7 = alt.Chart(df_sample_plot).mark_bar().encode(alt.X('deltaTime'), alt.Y('count()'),tooltip = ['mean(valsin)','deltaTime']).interactive()
chart7.show()
chart5 = alt.Chart(df_sample).mark_bar().encode(alt.X('claim_time'), alt.Y('mean(valsin)'))
chart5.show()
chart4 = alt.Chart(df_sample).mark_bar().encode(alt.X('PER_MOS_Social_Class_Code'), alt.Y('count()'))
chart4.show()

'''how valsin is distributed among the claim_time == 11?'''
chart5 = alt.Chart(df_sample_plot[df_sample_plot['claim_time']=='11']).mark_bar().encode(alt.X('valsin'), alt.Y('count()'))
chart5.show()
h = df_sample_plot[df_sample_plot['claim_time']=='11']

'''valsin distribution'''
chart_valsin = alt.Chart(df_sample).mark_bar().encode(alt.X('valsin'), alt.Y('count()', axis=alt.Axis( title='Frequency')))
#chart_valsin.show()
x = np.linspace (-5, 95, 500) 
y1 = stats.gamma.pdf(x, a=8, scale=0.5)
gamma = pd.DataFrame({'gamma_x': x, 'gamma_y': y1})
chart_gamma = alt.Chart(gamma).mark_line().encode(alt.X('gamma_x'), alt.Y('gamma_y'))
#chart_gamma.show()
df_sample['valsin'].mean()
chart_age = alt.Chart(df_sample).mark_bar().encode(alt.X('age'), alt.Y('count()', axis=alt.Axis( title='Frequency')))
#chart_age.show()
df_sample_kilo = df_sample[df_sample['kilowa']<700]
chart_kilowa = alt.Chart(df_sample_kilo).mark_bar().encode(alt.X('kilowa'), alt.Y('count()', axis=alt.Axis( title='Frequency')))
#chart_kilowa.show()
chart_valsin = alt.Chart(df_sample).mark_bar().encode(alt.X('valsin'), alt.Y('count()', axis=alt.Axis( title='Frequency')))
#chart_valsin.show()
alt.hconcat(alt.vconcat(chart_gamma, chart_valsin), alt.vconcat(chart_kilowa, chart_age))

#correlation heatmap
corr = df_sample.corr()
kot = corr[corr>=.6]
plt.figure(figsize=(12,8))
sns.heatmap(kot, cmap="Greens")

#QQ plot
measurements = df['valsin'] 
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()

#scatterplot 
sns.set()
cols = ['kilowa','valsin']
sns.pairplot(df[cols],size=10)
plt.show()

#age-valsin analysis
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
joint = sns.jointplot('age', 'valsin', data = df)



# One hot encoding - to convert categorical data to continuous
'''The challenge with CATEGORICAL VARIABLES is to find a suitable way to represent distances
 between variable categories and individuals in the factorial space. To overcome this problem, you can look for
 a non-linear transformation of each variable--whether it be nominal, ordinal, polynomial, or numerical--with optimal scaling'''
import statsmodels.api as sm
df_sample = df_checkpoint1
df_sample = df_sample[~df_sample['sincie'].isin(df_sample[df_sample.isna().any(axis=1)]['sincie'])]
df_sample = df_sample.sample(100000)
df_sample = df_sample.drop(columns = 'sincie')
n_samples = 100000
df_checkpoint2 = df_sample
df_sample = df_sample.sample(n_samples)#no dummies yes valsin
df_tot = pd.get_dummies(df_sample, columns = categorical_columns).sample(n_samples)#dummies and valsin included
y = df_tot['valsin']
X = df_tot.drop(columns = ['valsin'])
#Split in 80% train and 20% test set
train, test_df = train_test_split(df_tot, test_size = 0.20, random_state= 1984)
train_df, dev_df = train_test_split(train, test_size = 0.20, random_state= 1984)#valid. set is 20% of the training set
train_y = train_df['valsin']
test_y = test_df['valsin']
dev_y = dev_df['valsin']

train_df = train_df.drop(columns = 'valsin')
test_df = test_df.drop(columns = 'valsin')
dev_df = dev_df.drop(columns = 'valsin')
df_sample = df_sample.drop(columns = 'valsin')#no dummies no valsin
df_tot = df_tot.drop(columns = 'valsin')#dummies without valsin

'''– Validation set: A set of examples used to tune the parameters of a classifier,
 for example to choose the number of hidden units in a neural network.
'''
'''------------------------------------------DIMENSIONAL REDUCTION------------------------------------------------'''

'''------------------FA (scikit)---------------------'''

from sklearn.decomposition import FactorAnalysis
train_df_fa = train_df

train_df_fa =train_df_fa.loc[:,~train_df_fa.columns.duplicated()]

test_df_fa = test_df

test_df_fa = test_df_fa.loc[:,~test_df_fa.columns.duplicated()]

n_components = 16
transformer = FactorAnalysis(n_components=n_components, random_state=0)
#df_sample_ft_fa = transformer.fit(df_fa)
X_fa_red_train = transformer.fit_transform(train_df_fa)#both fit and dimensional reduction in the same function
X_fa_red_test = transformer.fit_transform(test_df_fa)
#components = transformer.fit(test_df_fa).components_.T

#score = transformer.score_samples(test_df_fa)
#dim_red_fa = transformer.transform(df_fa)#Apply dimensionality reduction to X using the model.Compute the expected mean of the latent variables

    
'''--------------------PCA(scikit)---------------------- '''
'''To interpret each principal components, examine the magnitude and direction of the coefficients 
for the original variables. The larger the absolute value of the coefficient, the more important the
 corresponding variable is in calculating the component. '''
#only continuous variables == 8 variables
continuous_columns = np.setdiff1d(df_sample.columns,categorical_columns)#Since the data were normalized, you can confirm that the principal components have variance 1.0
n_components = 6 
train_pca, test_df_pca = train_test_split(df_sample, test_size = 0.20, random_state= 1984)
train_df_pca, dev_df_pca = train_test_split(train_pca, test_size = 0.20, random_state= 1984)#valid. set is 20% of the training set
pca = PCA(n_components=n_components)
X_pca_red_train = pca.fit_transform(train_df_pca[continuous_columns])
X_pca_red_test = pca.fit_transform(test_df_pca[continuous_columns])
pca_components = pca.components_
# Dump components relations with features:
components_pca = pca.components_#eigenvectors
components_pca = pd.DataFrame(pca.components_, columns=continuous_columns ,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6'])
eigenvalues = pca.explained_variance_ #eigenvalues
pca.explained_variance_ratio_
'''---------------------MCA(mca - prince)---------------------'''

#categoricals have to be transformed in dummy( one hot). The documentation is super weak
train_mca, test_df_mca = train_test_split(df_sample, test_size = 0.20, random_state= 1984)
train_df_mca, dev_df_mca = train_test_split(train_mca, test_size = 0.20, random_state= 1984)#valid. set is 20% of the training set
train_df_mca = train_df_mca[categorical_columns]
test_df_mca = test_df_mca[categorical_columns]

train_df_mca = pd.get_dummies(train_df_mca, columns = categorical_columns)
test_df_mca = pd.get_dummies(test_df_mca, columns = categorical_columns)

n_components = 10

#mca 
import mca
mca_ind = mca.MCA(train_df_mca)#using Benzecri Correction
mca_ind_test =  mca.MCA(test_df_mca)#using Benzecri Correction

new_counts = pd.DataFrame(np.random.randint(0, 2, ( len(train_df_mca.index), len(train_df_mca.columns))))
new_counts_test = pd.DataFrame(np.random.randint(0, 2, ( len(test_df_mca.index), len(test_df_mca.columns))))

components_mca = mca_ind.fs_r_sup(new_counts,n_components)
components_mca_test = mca_ind_test.fs_r_sup(new_counts_test,n_components)
#factor scores of rows/columns?
components_mca = pd.DataFrame(components_mca , columns = ['PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16'])
components_mca_test = pd.DataFrame(components_mca_test , columns = ['PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16'])


# prince 
import prince
mca = prince.MCA(
      n_components=n_components,
      n_iter=3,
      copy=True,
      check_input=True,
      engine='auto',
      random_state=42
 )
mca = mca.fit(train_df_mca) # same as calling ca.fs_r_sup(df_new) for *another* test set.
components_fa = mca.V_
components_fa = pd.DataFrame(components_fa)


'''----------------------------Autoencoders : non linear transformation-----------------------'''
#------------------------------------Build the AutoEncoder------------------------------------
'BOTTLENECK AUTOENCODER''''
The activation function defines how the weighted sum of the input
is transformed into an output from a node or nodes in a layer of the network.
ELU:Exponential Linear Unit 
ELU becomes smooth slowly until its output equal to -α(end threshold for smoothing) whereas RELU sharply smoothes.
RELU:Rectified Linear Units.
it’s not linear and provides the same benefits as Sigmoid but with better performance.
ALTERNATIVE? A linear activation function
''
I AM MAKING A CLASS LABELS CLASSIFICATION: to learn a rule that computes the label from the attribute values (IS IT SUPERVISED LEARNING????)
Computes the cross-entropy loss between true labels and predicted labels.
Use this cross-entropy loss when there are only two label classes (assumed to be 0 and 1).
For each example, there should be a single floating-point value per prediction.
Cross-entropy builds upon the idea of entropy from information theory and calculates the 
number of bits required to represent or transmit an average event 
from one distribution compared to another distribution.
Binary Cross-Entropy: Cross-entropy as a loss function for a binary classification task.
Categorical Cross-Entropy: Cross-entropy as a loss function for a multi-class classification task.

Relative entropy(KL Divergence) :Average number of extra bits
to represent an event from Q instead of P.  

Adam optimization is a stochastic gradient descent method that is based on
adaptive estimation of first-order and second-order moments.
According to Kingma et al., 2014, the method is "computationally efficient,
has little memory requirement, invariant to diagonal rescaling of gradients, 
and is well suited for problems that are large in terms of data/parameters".
'''
train_df_nn = df_tot

train_x  = train_df_nn.loc[train_df_nn.index & train_df.index]
test_x  = train_df_nn.loc[train_df_nn.index & test_df.index]
dev_x  = train_df_nn.loc[train_df_nn.index & dev_df.index]

train_x =np.asarray(train_x).astype('float32')
dev_x =np.asarray(dev_x).astype('float32')
test_x = np.asarray(test_x).astype('float32')

# Choose size of our encoded representations (we will reduce our initial features to this number)
encoding_dim = 16 #final dimension I expect
# Define input layer( normalized continuous and one-hot dummies for categoricals)
input_data = Input(shape=(train_x.shape[1],))
# Define encoding layer
encoded = Dense(encoding_dim, activation='elu')(input_data)
# Define decoding layer
decoded = Dense(train_x.shape[1], activation='linear')(encoded)
# Create the autoencoder model
autoencoder = Model(input_data, decoded)
#Compile the autoencoder model

autoencoder.compile(optimizer='adamax',loss='mse') #why not KL divergence?
#Its possible to fix learning_rate and decay_rate of the optimizer algorithm.
#Fit to train set, validate with dev set and save to hist_auto for plotting purposes
hist_auto = autoencoder.fit(train_x, train_x,
                epochs = 30,
                batch_size=128,
                shuffle=True,
                validation_data =(dev_x, dev_x))


#autoencoder.get_layer('dense')
# Summarize loss function for the validation set
plt.figure()
plt.plot(hist_auto.history['loss'])
plt.plot(hist_auto.history['val_loss'])
plt.title('Autoencoder model loss YES NORMALISATION')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


'''
One Epoch is when an ENTIRE dataset is passed forward and backward through 
the neural network only ONCE.
Loss function is not becoming flat: I can try a higher learning rate,
 and more epochs with an early stopping method if you have enough data.
'''
# Create a separate model (encoder) in order to make encodings (first part of the autoencoder model)
encoder = Model(input_data, encoded)
# Create a placeholder for an encoded input
encoded_input = Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
#Encode data set from above using the encoder



'''MODEL COMPARISON: GAMMA DEVIANCE per test fold'''

'''----------------------------3. MODELLING---------------------------''' 

'''--------------------------------------------GAMMA GLM-----------------------------------'''

'''------------------------------------PCA(6) + MCA(10) FEATURES-----------------------------------------'''
import statsmodels.api as sm
Xtrain = pd.DataFrame(X_pca_red_train, columns = ['PC1','PC2','PC3','PC4','PC5','PC6'])
Xtrain = pd.concat([Xtrain, components_mca], axis = 1)
Xtrain['valsin'] = train_y.reset_index(drop=True)

Xtest = pd.DataFrame(X_pca_red_test, columns = ['PC1','PC2','PC3','PC4','PC5','PC6'])
Xtest = pd.concat([Xtest, components_mca_test], axis = 1)
Xtest['valsin'] = test_y.reset_index(drop=True)

import statsmodels.formula.api as smf
gamma_model = smf.glm(formula='valsin ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 +PC14 + PC15 + PC16',
                      data=Xtrain, family=sm.families.Gamma(sm.families.links.log))
gamma_results = gamma_model.fit()
print(gamma_results.summary())

y_pred = gamma_results.predict(Xtest)
mean_gamma_deviance(test_y, y_pred)#0.7051322112395837
string = gamma_results.summary().as_latex()
#interdependence


'''-------------------------------- Factor Analysis(16) -------------------------------------------'''
Xtrain = pd.DataFrame(X_fa_red_train, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12',
                                                 'PC13','PC14','PC15','PC16'])
Xtest = pd.DataFrame(X_fa_red_test, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12',
                                                 'PC13','PC14','PC15','PC16'])

Xtrain['valsin'] = pd.DataFrame(train_y).reset_index(drop=True)

import statsmodels.formula.api as smf
gamma_model = smf.glm(formula='valsin ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 +PC14 + PC15 + PC16',
                      data=Xtrain, family=sm.families.Gamma(sm.families.links.log))
gamma_results = gamma_model.fit()
print(gamma_results.summary())

y_pred = gamma_results.predict(Xtest)
mean_gamma_deviance(test_y, y_pred)#0.7275129978493636


'''------------------------------------AUTOENCODERS COMPONENTS(16)----------------------------'''
encoded_test_x = encoder.predict(test_x)
encoded_train_x = encoder.predict(train_x)
decoded_pred = decoder.predict(encoded_test_x)
Xtrain = pd.DataFrame(encoded_train_x, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12',
                                                 'PC13','PC14','PC15','PC16'])
Xtest = pd.DataFrame(encoded_test_x, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12',
                                                 'PC13','PC14','PC15','PC16'])
Xtrain['valsin'] = pd.DataFrame(train_y).reset_index(drop=True)

import statsmodels.formula.api as smf
gamma_model = smf.glm(formula='valsin ~ PC1 + PC2 + PC3 +PC4 + PC5 + PC6 + PC7+ PC8+ PC9+ PC10+ PC11+ PC12+ PC13+ PC14+ PC15+ PC16',
                      data=Xtrain, family=sm.families.Gamma(sm.families.links.log))
gamma_results = gamma_model.fit()

y_pred = pd.Series(gamma_results.predict(Xtest))
test_y = pd.Series(test_y)
mean_gamma_deviance(test_y, y_pred)#0.6865254093350335

gamma_results.null_deviance 
gamma_results.deviance
'''
test_y[test_y == 0].index#11087
test_y = test_y.drop(test_y.index[11087])
y_pred = y_pred.drop(y_pred.index[11087])

#y_pred = [0 if y_pred_ < 0 else y_pred_ for y_pred_ in y_pred]
#test_y = [0 if test_y < 0 else test_y_ for test_y_ in test_y]
'''

mean_gamma_deviance(test_y, y_pred)#0.6865254093350335
#0.6656904849909232 with 70k data and 30 epochs



'''------------------------------------GBM------------------------------------------------'''
'''
df_tot.to_csv('Z:\RW_FOLDER\projects\cca_data\claudio_thesis\df_sample_for_GBM.csv',index = False)  
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
h2o.init()

cars = h2o.import_file("Z:\RW_FOLDER\projects\cca_data\claudio_thesis\df_sample_for_GBM.csv")
predictors = ['coll39', 'manexp','carbur', 'manexp','effrac', 'antivo','degatx',
                       'degat0','degat1','degat2','degat3','degat4','degat5','degat6','degat7','degat8','degat9',
                       'degat_tot', 'vehimmo', 'contecA','contecB', 'contecR','contecN', 'contec1','contec2','contec3',
                       'contec4', 'region', 'province','bodyshop_postal_code', 'camion',
                       'autoclosure','gamme1', 'gamme2', 'gamme3', 'gamme4', 'gamme41','gamme5','gamme6','cat1',
                       'cat2', 'cat3', 'cat4', 'cat5','cat6', 'cat7','marque', 'modell', 'claim_branch_cd', 
                       'circumstance_cd','claim_responsibility_rate','PER_MOS_Social_Class_Code','PER_MOS_Urbanization_Lvl_cd',
                      'PER_MOS_Net_Income_Average', 'private_corporate_cd_C','role_cd','PER_langage_cd_C', 
                      'PER_gender','PER_postal_cd','PER_occupation','kiloml','kilowa','pieces','duration_mission_sent_closed',
                    'claim_time','valass']
response = "valsin"
train, valid = cars.split_frame(ratios = [.8], seed = 1234)

# try using the distribution parameter:
# Initialize and train a GBM
cars_gbm = H2OGradientBoostingEstimator(distribution = "gamma", seed = 1234)
cars_gbm.train(x = predictors, y = response, training_frame = train, validation_frame = valid)
cars.gbm.predict
# print the MSE for the validation data
cars_gbm.mse(valid=True)
'''
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from catboost import Pool

train_g, X_test = train_test_split(df_sample, test_size = 0.20, random_state= 1984)
X_train, dev_gbm = train_test_split(train_g, test_size = 0.20, random_state= 1984)#valid. set is 20% of the training set

 
X_train[['claim_branch_cd','claim_time']] = X_train[['claim_branch_cd','claim_time']].astype(int)
X_test[['claim_branch_cd','claim_time']] = X_test[['claim_branch_cd','claim_time']].astype(int)

cats=list(X_train.dtypes.where((X_train.dtypes=='object')).dropna().index)
# Contrainte monotone sur trois variables.
cols_increasing=['degat_tot','kilowa']
num =[1 if x in cols_increasing else 0 for x in X_train.columns]

clf=CatBoostRegressor(iterations=500,learning_rate=0.1,max_depth=10,eval_metric='Tweedie:variance_power=1.99', loss_function='Tweedie:variance_power=1.99',
                        random_seed = 23,monotone_constraints=num,thread_count=2,early_stopping_rounds=10,
                        cat_features=cats)

g=clf.fit(X_train,train_y,use_best_model=False,verbose=True)#_train, train_y,eval_set=(X_test,test_y)


y_pred=g.predict(X_test)
#mean_absolute_error(test_y, y_pred)
mean_gamma_deviance(test_y, y_pred)#0.683

'''---------------------------------Plots GBM---------------------------------'''
feature_score = pd.DataFrame(list(zip(X_train.dtypes.index, clf.get_feature_importance(Pool(X_train, label=train_y, cat_features=cats)))),
                columns=['Feature','Score'])
feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')
plt.rcParams["figure.figsize"] = (12,7)
ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)
ax.set_xlabel('')

rects = ax.patches

labels = feature_score['Score'].round(2)

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')
plt.show()
