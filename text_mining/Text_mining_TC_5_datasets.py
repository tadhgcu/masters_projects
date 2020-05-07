
# coding: utf-8

# **Optional Text Mining Task**
# 
# Perform the previous task but this time using **5 datasets**
# 
#     Part I. Construction of an automatic classifier
#     Part II. Construction of a clustering of biology documents
# 
# _**Note**: activate nbextensions "Table of contents"_

# # Imports

# In[22]:


import pandas as pd
import numpy as np
import string
import re
import pickle
import csv
import matplotlib.pyplot as plt

from sklearn import svm, naive_bayes, neighbors, tree, preprocessing, pipeline, metrics
from sklearn.cluster import KMeans, ward_tree 
from sklearn.decomposition import TruncatedSVD 
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer


get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the data

# In[68]:


#all data in one file
data = ["./Alzheimer_abstracts.tsv", "./Bladder_Cancer_abstracts.tsv", "./Breast_Cancer_abstracts.tsv",
       "./Cervical_Cancer_abstracts.tsv", "./negative_abstracts.tsv"]
cat = ["alz", "bladder", "breast", "cervical", "neg"]
 
corpus = []
number_list = []
text_list = []
abstract_list = []
category_list = []

for i in range(0,5,1):
    file = data[i]
    with open(file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for line in reader:
            corpus.append(line)
            number_list.append(line[0]) #number
            text_list.append(line[1]) #titles
            abstract_list.append(line[2]) #only abstracts
            category_list.append(cat[i]) #category      
   


# ## Make into a dataframe

# In[69]:


#easier to work with a dataframe
data = pd.DataFrame(corpus)
data['Doc_number'] = pd.DataFrame(number_list)
data['Titles'] = pd.DataFrame(text_list)
data['Abstracts'] = pd.DataFrame(abstract_list)
data['Type'] = pd.DataFrame(category_list)
data.drop(data.columns[[0,1,2]], axis=1, inplace=True)
data.head()


# In[70]:


class_column = 'Type'
classes_names = data['Type'].unique()
attribute_columns = list(data.columns)
attribute_columns.remove(class_column)

print(class_column)
print(classes_names)
print(attribute_columns)


# # Part I - Construction of an automatic classifier (5 datasets)

# ## Text Processing

# ### Clean the data

# In[71]:


#clean the data
stemmer = SnowballStemmer('english')
words = stopwords.words("english")

#now clean it all, removing stop words, caps...
data['cleaned_Abstracts'] = data['Abstracts'].apply(
    lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
data.head()


# ### Split the data into test and train

# In[72]:


#split the data into test and train

X = data['cleaned_Abstracts']
Y = data['Type']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


# ### Parameters

# In[73]:


#default parameters 
min_df = 0.2
max_df = 0.8
stop_words = 'english' 
max_features = 200
norm = 'l1'
n_components = 20

#Parameters changed
#Tfidf
#min_df=0.05 # 0.05, 0.15, 0.2
#max_df=0.3 #0.3, 0.6, 0.9
#stop_words = 'english' # 'english', 'None'
#max_features= 500 # 200, 500, 2000, 5000
#norm = 'l1' # 'l1' 'l2' 'None'
#LSA
#n_components = 10 # 20 30


# ### Make Piplines to process training data

# In[74]:


classifier_NB = Pipeline([('Tfidf', TfidfVectorizer(min_df=min_df, 
                                                 max_df=max_df, 
                                                 stop_words = stop_words,
                                                 max_features= max_features,
                                                 norm = norm,
                                                 lowercase=False)),
                       ('LSA', TruncatedSVD(n_components = n_components)),
                       ('scaler', StandardScaler()),
                       ('NB', GaussianNB())])

classifier_SVC = Pipeline([('Tfidf', TfidfVectorizer(min_df=min_df, 
                                                 max_df=max_df, 
                                                 stop_words = stop_words,
                                                 max_features= max_features,
                                                 norm = norm,
                                                 lowercase=False)),
                       ('LSA', TruncatedSVD(n_components = n_components)),
                       ('scaler', StandardScaler()),
                       ('SVC', SVC(C = 1))])

classifier_KNN = Pipeline([('Tfidf', TfidfVectorizer(min_df=min_df, 
                                                 max_df=max_df, 
                                                 stop_words = stop_words,
                                                 max_features= max_features,
                                                 norm = norm,
                                                 lowercase=False)),
                       ('LSA', TruncatedSVD(n_components = n_components)),
                       ('scaler', StandardScaler()),
                       ('KNN', KNeighborsClassifier(n_neighbors = 10))])

classifier_tree = Pipeline([('Tfidf', TfidfVectorizer(min_df=min_df, 
                                                 max_df=max_df, 
                                                 stop_words = stop_words,
                                                 max_features= max_features,
                                                 norm = norm,
                                                 lowercase=False)),
                       ('LSA', TruncatedSVD(n_components = n_components)),
                       ('scaler', StandardScaler()),
                       ('tree', DecisionTreeClassifier(max_depth= 7))])


# ## Test the classifiers

# ### Run Pipelines over training data

# In[75]:


Model_tree = classifier_tree.fit(X_train, y_train)
Model_NB = classifier_NB.fit(X_train, y_train)
Model_SVC = classifier_SVC.fit(X_train, y_train)
Model_KNN = classifier_KNN.fit(X_train, y_train)


# ### Test the accuracy of each model on the test data

# In[76]:


model_list = [Model_tree, Model_NB, Model_SVC, Model_KNN]
model_name_list = ["Model_tree", "Model_NB", "Model_SVC", "Model_KNN"]
ytest = np.array(y_test)
count = 0
for i in model_list:
    accuracy = accuracy_score(i.predict(X_test), ytest)
    print(model_name_list[count],"=",accuracy)
    count = count+1


# ### Print the confusion matrix for each classifier

# In[77]:


ytest = np.array(y_test)

# confusion matrix and classification report(precision, recall, F1-score)
print(classification_report(ytest, Model_NB.predict(X_test)))
print("Confusion matrix:")
print(confusion_matrix(ytest, Model_NB.predict(X_test)))


# In[78]:


print(classification_report(ytest, Model_tree.predict(X_test)))
print("Confusion matrix:")
print(confusion_matrix(ytest, Model_tree.predict(X_test)))


# In[79]:


print(classification_report(ytest, Model_SVC.predict(X_test)))
print("Confusion matrix:")
print(confusion_matrix(ytest, Model_SVC.predict(X_test)))


# In[80]:


print(classification_report(ytest, Model_KNN.predict(X_test)))
print("Confusion matrix:")
print(confusion_matrix(ytest, Model_KNN.predict(X_test)))


# # Part II - Clustering the documents

# ## Ensure there is no missing data in the dataset

# In[81]:


#confirm there are no missing values in the data as Kmeans doesnt perform well with missing data
print("*****In the Cleaned Abstracts*****")
print(data['cleaned_Abstracts'].isna().sum())
print("*****In the Type*****")
print(data['Type'].isna().sum())


# In[82]:


#No training data for clustering
y = np.array(data[class_column])
X = np.array(data['cleaned_Abstracts'])


# ## Parameters 

# In[83]:


max_features=500
norm = 'l1'
n_components = 20


# ## Code for each clustering classifier

# In[85]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import string
def preprocessing(line):
    line = line.lower()
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    return line
tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocessing,
                                   max_features=max_features, #1
                                   norm = norm, #2
                                   lowercase=False)
tfidf = tfidf_vectorizer.fit_transform(X)
LSA = TruncatedSVD(n_components = n_components, #3
                   random_state = 0)
LSAX = LSA.fit_transform(tfidf) #4
scaler = StandardScaler()
scaX = scaler.fit_transform(LSAX)
normalizer = Normalizer() #5
NX = normalizer.fit_transform(scaX)


# ## Train the Models

# ### K-means

# In[86]:


kmeans = KMeans(n_clusters=5, init='k-means++',n_init = 1, max_iter=300, random_state=2).fit(NX)

# y is the Type (neg, pos)
unique_y = np.unique(y)
ids_clusters = kmeans.labels_
for i in np.unique(ids_clusters):
    inds = (np.where(np.array(ids_clusters) == i))[0]
    print('\033[1m'+'- Cluster %d' % i + '\033[0m')
    print('  %g%% of total patterns' % (100*len(inds)/len(ids_clusters)))
    for real_class in unique_y:
        clustered = (list(y[inds])).count(real_class)
        total = len(y)
        print(real_class,":", (clustered/total)*100 )


# ### Hierarchical clustering using ward

# In[87]:


from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# Hierarchichal clustering, single-linkage:
ward_cluster = linkage(NX, 'ward')

unique_y = np.unique(y)
ids_clusters = fcluster(ward_cluster, 5, # number of final clusters
                    criterion='maxclust') - 1
for i in np.unique(ids_clusters):
    inds = (np.where(np.array(ids_clusters) == i))[0]
    print('\033[1m'+'- Cluster %d' % i + '\033[0m')
    print('  %g%% of total patterns' % (100*len(inds)/len(ids_clusters)))
    for real_class in unique_y:
        clustered = (list(y[inds])).count(real_class)
        total = len(y)
        print(real_class, ":" ,(clustered/total)*100 )
    print()


# In[88]:


# Plot the dendrogram:
plt.figure(figsize=(20, 10))
dendrogram(ward_cluster, leaf_rotation=0, truncate_mode='lastp', p=100)
plt.grid(True)

