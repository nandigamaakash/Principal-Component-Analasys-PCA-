#> %reset  #clear all variables
"""
@author: A. Ben Hamza
"""

'''Uncomment line 14 and line 26 for Google Collab'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#use seaborn plotting style defaults
import seaborn as sns; sns.set()
#from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from scipy.stats import beta
from scipy.stats import f
from sklearn.neighbors import KNeighborsClassifier

#upload files into Colaboratory
#uploaded = files.upload()

#read cvs file into dataframe
data = pd.read_csv('Pizza.csv', index_col=None)
print(data.head())
data.head()
np.random.seed(48)

#m,n=df.shape #size of data
#X = df.ix[:,0:n].values # Feature matrix
#from sklearn.preprocessing import StandardScaler
#X = StandardScaler().fit_transform(X) #normalize data

#remove target value from the dataset
df = data.drop(['brand','id'], axis=1)
#normalize data
df = (df - df.mean())/df.std()
# Displaying DataFrame columns.
df.columns
# Some basic information about each column in the DataFrame 
df.info()
print(df.columns)
Y=data["brand"]
df = df.dropna()

#observations and variables
observations = list(df.index)
variables = list(df.columns)

#visualisation of the data using a box plot
ax = sns.boxplot(data=df, orient="v", palette="Set2")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

#pairplot
plt.figure()
sns.pairplot(df)
plt.title('Pairplot')

#Covariance
dfc = df - df.mean() #centered data
plt. figure()
ax = sns.heatmap(dfc.cov(), center=0,cmap='RdYlGn_r', linewidths=0.5, annot=True, 
            cbar=True, square=True)
plt.yticks(rotation=0)
ax.tick_params(labelbottom=False,labeltop=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.title('Covariance matrix')

#Principal component analysis
pca = PCA()
pca.fit(df)
Z = pca.fit_transform(df)

plt. figure()
plt.scatter(Z[:,0], Z[:,1], c='r')
plt.xlabel('$Z_1$')
plt.ylabel('$Z_2$')

#Eigenvectors
A = pca.components_.T 
plt. figure()
plt.scatter(A[:,0],A[:,1],c='r')
plt.xlabel('$A_1$')
plt.ylabel('$A_2$');
for label, x, y in zip(variables, A[:, 0], A[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-2, 2),
        textcoords='offset points', ha='right', va='bottom')

plt. figure()
plt.scatter(A[:, 0],A[:, 1],marker='o',c=A[:, 2],s=A[:, 3]*500,
    cmap=plt.get_cmap('Spectral'))
for label, x, y in zip(variables,A[:, 0],A[:, 1]):
    plt.annotate(label,xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

#Eigenvalues
Lambda = pca.explained_variance_

#Scree plot
plt. figure()
x = np.arange(len(Lambda)) + 1
plt.plot(x,Lambda/Lambda.sum(), 'ro-', lw=2)
plt.xticks(x, [""+str(i) for i in x], rotation=0)
plt.xlabel('Number of components')
plt.ylabel('Explained variance')

(Lambda[1]+Lambda[0])/Lambda.sum()
print("Eigen values",Lambda)

#Explained variance
ell = pca.explained_variance_ratio_
plt. figure()
ind = np.arange(len(ell))
plt.bar(ind, ell, align='center', alpha=0.5)
plt.plot(np.cumsum(ell))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

#Biplot
# 0,1 denote PC1 and PC2; change values for other PCs
A1 = A[:,0] 
A2 = A[:,1]
A3 = A[:,2]
Z1 = Z[:,0] 
Z2 = Z[:,1]
Z3 = Z[:,2]
print("Matrix A",A)

fig, ax = plt.subplots()

for i in range(len(A1)):
# arrows project features as vectors onto PC axes
    ax.arrow(0, 0, A1[i]*max(Z1), A2[i]*max(Z2),
              color='black', width=0.00005, head_width=0.0025)
    ax.text(A1[i]*max(Z1)*1.2, A2[i]*max(Z2)*1.2, variables[i], color='black')

for i in data['brand'].unique():
# circles project documents (ie rows from csv) as points onto PC axes
#colors=['red','green','purple']
  ax.scatter(Z1[data['brand']==i], Z2[data['brand']==i], marker='o',label=str(i))
 #   plt.text(Z1[i]*1.2, Z2[i]*1.2, observations[i], color='b')
legend = ax.legend(shadow=False, ncol=3, bbox_to_anchor=(0.85, -0.1))

plt.show()

comps = pd.DataFrame(A,columns = variables)
sns.heatmap(comps,cmap='RdYlGn_r', linewidths=0.5, annot=True, 
            cbar=True, square=True)
ax.tick_params(labelbottom=False,labeltop=True)
plt.title('Principal components')





#Hotelling's T2 test
alpha = 0.05
p=Z.shape[1]
n=Z.shape[0]

UCL=((n-1)**2/n )*beta.ppf(1-alpha, p / 2 , (n-p-1)/ 2)
UCL2=p*(n+1)*(n-1)/(n*(n-p) )*f.ppf(1-alpha, p , n-p)
Tsquare=np.array([0]*Z.shape[0])
for i in range(Z.shape[0]):
  Tsquare[i] = np.matmul(np.matmul(np.transpose(Z[i]),np.diag(1/Lambda) ) , Z[i])

fig, ax = plt.subplots()
ax.plot(Tsquare,'-b', marker='o', mec='y',mfc='r' )
ax.plot([UCL for i in range(len(Z1))], "--g", label="UCL")
plt.ylabel('Hotelling $T^2$')
legend = ax.legend(shadow=False, ncol=4, bbox_to_anchor=(0.85, -0.1))

fig.show()

#out of control points
print (np.argwhere(Tsquare>UCL))


#Control Charts for Principle Components 
fig, ax = plt.subplots()
ax.plot(Z1,'-b', marker='o', mec='y',mfc='r' , label="Z1")
ax.plot([3*np.sqrt(Lambda[0]) for i in range(len(Z1))], "--g", label="UCL")
ax.plot([-3*np.sqrt(Lambda[0]) for i in range(len(Z1))], "--g", label='LCL')
ax.plot([0 for i in range(len(Z1))], "-", color='black',label='CL')
plt.ylabel('$Z_1$')
legend = ax.legend(shadow=False, ncol=4, bbox_to_anchor=(0.85, -0.1))

fig.show()

#classification

logisticRegr = LogisticRegression(solver='lbfgs')
scoring=['accuracy']
scores_lr_full_data = cross_validate(logisticRegr, df, Y,cv=5, scoring=scoring)
scores_lr_Z = cross_validate(logisticRegr, Z, Y,cv=5, scoring=scoring)
scores_lr_Z12 = cross_validate(logisticRegr, Z[:,:2], Y,cv=5, scoring=scoring)
print(scores_lr_full_data)
print(scores_lr_Z12)
gnb = GaussianNB()
scoring=['accuracy']
scores_gnb_full_data = cross_validate(gnb, df, Y,cv=5, scoring=scoring)
scores_gnb_Z = cross_validate(gnb, Z, Y,cv=5, scoring=scoring)
scores_gnb_Z12 = cross_validate(gnb, Z[:,:2], Y,cv=5, scoring=scoring)
scores_dict={}
for i in ['fit_time','test_accuracy']:
  scores_dict["lr_full_data " + i ]=scores_lr_full_data[i]
  scores_dict["lr_Z  " + i ]=scores_lr_Z[i]
  scores_dict["lr_Z12 " + i ]=scores_lr_Z12[i]
  scores_dict["gnb_full_data " + i ]=scores_gnb_full_data[i]
  scores_dict["gnb_Z " + i ]=scores_gnb_Z[i]
  scores_dict["gnb_Z12 " + i ]=scores_gnb_Z12[i]

scores_data=pd.DataFrame(scores_dict)
print(scores_data)

#SupportVectorMachine
svm = SVC() 
scores_svm_full_data = cross_validate(svm, df, Y,cv=5, scoring=scoring)
scores_svm_Z = cross_validate(svm, Z, Y,cv=5, scoring=scoring)
scores_svm_Z12 = cross_validate(svm, Z[:,:2], Y,cv=5, scoring=scoring)
#print(scores_svm_Z)
#print(scores_svm_Z12)

#KNN
knn = KNeighborsClassifier(n_neighbors=13)
scores_knn_full_data = cross_validate(knn, df, Y,cv=5, scoring=scoring)
scores_knn_Z = cross_validate(knn, Z, Y,cv=5, scoring=scoring)
scores_knn_Z12 = cross_validate(knn, Z[:,:4], Y,cv=5, scoring=scoring)
#print(scores_knn_Z)
#print(scores_knn_Z12)
scores_dict={}
for i in ['fit_time','test_accuracy']:
  scores_dict["lr_full_data " + i ]=scores_lr_full_data[i]
  scores_dict["lr_Z  " + i ]=scores_lr_Z[i]
  scores_dict["lr_Z12 " + i ]=scores_lr_Z12[i]
  scores_dict["svm_full_data " + i ]=scores_svm_full_data[i]
  scores_dict["svm_Z " + i ]=scores_svm_Z[i]
  scores_dict["svm_Z12 " + i ]=scores_svm_Z12[i]

scores_data=pd.DataFrame(scores_dict)
print(scores_data)
scores_dict={}
for i in ['fit_time','test_accuracy']:
  scores_dict["lr_full_data " + i ]=scores_lr_full_data[i]
  scores_dict["lr_Z  " + i ]=scores_lr_Z[i]
  scores_dict["lr_Z12 " + i ]=scores_lr_Z12[i]
  scores_dict["knn_full_data " + i ]=scores_knn_full_data[i]
  scores_dict["knn_Z " + i ]=scores_knn_Z[i]
  scores_dict["knn_Z12 " + i ]=scores_knn_Z12[i]

scores_data=pd.DataFrame(scores_dict)
print(scores_data)


#discplay coefficients
X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2)
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
score = logisticRegr.score(X_test, y_test)
print("before PCA", score)
coefficient_full = logisticRegr.coef_

Z_train, Z_test, yz_train, yz_test = train_test_split(Z, Y, test_size=0.2)
logisticRegr_z = LogisticRegression()
logisticRegr_z.fit(Z_train, yz_train)
score_z = logisticRegr_z.score(Z_test, yz_test)
print("After PCA",score_z)
coefficient_PCA = logisticRegr_z.coef_
np.around(coefficient_full, decimals=2)

np.around(coefficient_PCA, decimals=2)