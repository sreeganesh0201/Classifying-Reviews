
# coding: utf-8

# In[7]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import os
print(os.listdir("C:/Users/sreeg/OneDrive - Oklahoma State University/projects/sentiment analysis"))


# In[10]:


import pandas
df_review = pandas.read_csv('C:/Users/sreeg/OneDrive - Oklahoma State University/projects/sentiment analysis/amazon_alexa.tsv', sep='\t')
df_review.head()


# In[11]:


df_review.describe()


# In[12]:


df_review.groupby('rating').describe()


# In[13]:


df_review['length'] = df_review['verified_reviews'].apply(len)
df_review.head()


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


df_review['length'].plot(bins=70, kind='hist')


# In[21]:


df_review.hist(column='length', by='feedback', bins=50,figsize=(10,4))


# In[22]:


import numpy as np
import matplotlib.pyplot as plt
# Importing the dataset
dataset = pd.read_csv('C:/Users/sreeg/OneDrive - Oklahoma State University/projects/sentiment analysis/amazon_alexa.tsv', delimiter = '\t', quoting = 3)


# In[34]:


# Cleaning the texts
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,3150):
    review = re.sub('[^a-zA-Z]', ' ', dataset['verified_reviews'][i] )
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
review


# In[35]:


corpus


# In[38]:


# creating the Bag of words Model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
Y=dataset.iloc[:,4].values


# In[39]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


# In[41]:


# Fitting Random Forest classifier with 100 trees to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


# In[42]:


# Predicting the Test set results
Y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# In[43]:


cm

