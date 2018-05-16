
# coding: utf-8

# ## Machine Learnning - Previsão banco de dados de Câncer

# In[1]:

from sklearn.datasets import load_breast_cancer


# In[2]:

cancer = load_breast_cancer()


# In[3]:

print("cancer.keys(): \n{}".format(cancer.keys()))


# In[4]:

print(cancer.feature_names)


# In[5]:

print(cancer.data)


# In[6]:

print(cancer.DESCR)


# In[7]:

print(cancer.target_names)


# In[20]:

X = cancer.data
y = cancer.target


# In[21]:

print(X.shape)
print(y.shape)


# In[39]:

from sklearn.neighbors import KNeighborsClassifier
knm = KNeighborsClassifier(n_neighbors=1)
knm.fit(X, y)
y_pred = knm.predict(X)
print(metrics.accuracy_score(y, y_pred))


# In[40]:

from sklearn.neighbors import KNeighborsClassifier
knm = KNeighborsClassifier(n_neighbors=5)
knm.fit(X, y)
y_pred = knm.predict(X)
print(metrics.accuracy_score(y, y_pred))


# In[42]:

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X, y)
logreg.predict(X)
y_pred = logreg.predict(X)
from sklearn import metrics
print(metrics.accuracy_score(y, y_pred))


# In[43]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


# In[48]:

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[47]:

knm = KNeighborsClassifier(n_neighbors=5)
knm.fit(X_train, y_train)
y_pred = knm.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[49]:

knm = KNeighborsClassifier(n_neighbors=1)
knm.fit(X_train, y_train)
y_pred = knm.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[50]:

k_range = list(range(1, 26))
scores = []
for k in k_range:
    knm = KNeighborsClassifier(n_neighbors=k)
    knm.fit(X_train, y_train)
    y_pred = knm.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))


# In[51]:

print(scores)


# In[52]:

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

plt.plot(k_range, scores)
plt.xlabel("Value of K for KNM")
plt.ylabel("Testing accuracy")


# ## RESPOSTAS

# ## 1) Test entire model: 

# ## knm(1) -> 100%
# ## knm(5) -> 94,73%
# ## LogReg -> 95,96%

# ## 2) Train/test Split:

# ##  knm(1) -> 90,35%
# ## knm(5) -> 90,79%
# ## LogReg - > 90,79%

# ## 3) Gráfico de Tunning knm (1-25)

# In[56]:

plt.plot(k_range, scores)


# ## 4) Qual o melhor modelo com a melhor acurácia?

# ## Knm(3)

# In[ ]:



