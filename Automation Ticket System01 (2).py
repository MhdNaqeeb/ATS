#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re, nltk, spacy, string
import en_core_web_sm
nlp = en_core_web_sm.load()


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
import os
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv(r"C:\Users\Mhd Naqeeb\Downloads\data.csv")
data.head(20)


# In[4]:


#print the column names
data.columns


# In[5]:


#Assign new column names
data = data[['source.complaint_what_happened','source.product','source.sub_produc']]
data.head()


# In[6]:


# Rename
data = data.rename(columns={'source.complaint_what_happened': 'complaint_text', 'source.product': 'category','source.sub_produc': 'sub_category'})
data.head()


# In[7]:


data = data.rename(columns={'source.complaint_what_happened': 'complaint_text', 'source.product': 'category','source.sub_produc': 'sub_category'})
data.head()


# In[8]:


#nun complaints
data.complaint_text.isnull().sum()


# In[9]:


#  empty complaints
len(data[data['complaint_text']==''])


# In[10]:


#replace empty values
data[data['complaint_text']==''] = np.nan
data.complaint_text.isnull().sum()


# In[11]:


data = data[~data['complaint_text'].isnull()]
data.complaint_text.isnull().sum()


# In[12]:


import os
import nltk
import nltk.corpus
import re, nltk, spacy, string
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# In[13]:


def clean_text(text):
    text = text.lower()  # Make the text lowercase
    text = re.sub('\[.*\]','', text).strip() # Removing text in square brackets
    text = text.translate(str.maketrans('', '', string.punctuation)) # Removing punctuation
    text = re.sub('\S*\d\S*\s*','', text).strip()  # Rstringemoving words containing numbers
    return text.strip()


# In[14]:


data.complaint_text = data.complaint_text.apply(lambda x: clean_text(x))
data.complaint_text.head()


# In[15]:


stopwords = nlp.Defaults.stop_words
def lemmatizer(text):
    doc = nlp(text)
    sent = [token.lemma_ for token in doc if not token.text in set(stopwords)]
    return ' '.join(sent)


# In[16]:


data['lemma'] =  data.complaint_text.apply(lambda x: lemmatizer(x))
data.head()


# In[17]:


data_clean = data[['complaint_text','lemma','category']]
data_clean.head()


# In[18]:


def extract_pos_tags(text):
    doc = nlp(text)
    sent = [token.text for token in doc if token.tag_ == 'NN']
    return ' '.join(sent)


# In[19]:


data_clean['complaint_POS_removed'] =  data_clean.lemma.apply(lambda x: extract_pos_tags(x))
data_clean.head()


# In[20]:


plt.bar(data['complaint_text'], data['lemma'])
plt.xlabel('complaint_text')
plt.ylabel('lemma')
plt.title('lemma')
plt.show()


# In[21]:


plt.figure(figsize=(10,6))
doc_lens = [len(d) for d in data_clean.complaint_POS_removed]
plt.hist(doc_lens, bins = 50)


# In[22]:


from wordcloud import WordCloud
wordcloud = WordCloud(stopwords=stopwords,max_words=40).generate(str(data_clean.complaint_POS_removed))
print(wordcloud)
plt.figure(figsize=(10,6))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[23]:


#Removing PRON
data_clean['Complaint_clean'] = data_clean['complaint_POS_removed'].str.replace('-PRON-', '')
data_clean = data_clean.drop(['complaint_POS_removed'],axis = 1)
def get_top_n_bigram(text, ngram=1, top=None):
    vec = CountVectorizer(ngram_range=(ngram, ngram), stop_words='english').fit(text)
    bag_of_words = vec.transform(text)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:top]
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pprint import pprint
top_30_unigrams = get_top_n_bigram(data_clean.Complaint_clean,ngram=1, top=30)
top_30_bigrams = get_top_n_bigram(data_clean.Complaint_clean,ngram=2, top=30)
top_30_trigrams = get_top_n_bigram(data_clean.Complaint_clean,ngram=3, top=30)
print('Top 10 unigrams:\n')
top_30_unigrams[:10]


# In[24]:


data1 = pd.DataFrame(top_30_unigrams, columns = ['unigram' , 'count'])
plt.figure(figsize=(12,6))
fig = sns.barplot(x=data1['unigram'], y=data1['count'])
plt.xticks(rotation = 80)
plt.show()


# In[25]:


print('Top 10 bigrams:\n')
top_30_bigrams[:10]


# In[26]:


#Plot graph for the top 30 words in the bigram frequency
data2 = pd.DataFrame(top_30_bigrams, columns = ['bigram' , 'count'])
plt.figure(figsize=(12,6))
fig = sns.barplot(x=data2['bigram'], y=data2['count'])
plt.xticks(rotation = 80)
plt.show()


# In[27]:


#Print the top 10 words in the trigram frequency
print('Top 10 trigrams:\n')
top_30_trigrams[:10]


# In[28]:


#Plot graph for the top 30 words in the trigram frequency
data3 = pd.DataFrame(top_30_trigrams, columns = ['trigram' , 'count'])
plt.figure(figsize=(12,6))
fig = sns.barplot(x=data3['trigram'], y=data3['count'])
plt.xticks(rotation = 80)
plt.show()


# In[29]:


data_clean['Complaint_clean'] = data_clean['Complaint_clean'].str.replace('xxxx','')
#All masked texts has been removed
data_clean.head()


# In[30]:


tfidf = TfidfVectorizer(min_df=2, max_df=0.95, stop_words='english')


# In[31]:


dtm = tfidf.fit_transform(data_clean.Complaint_clean)


# In[32]:


tfidf.get_feature_names_out()


# In[33]:


tfidf.get_feature_names_out()[:10]


# In[34]:


len(tfidf.get_feature_names_out())


# Modleing 

# In[35]:


from sklearn.decomposition import NMF


# In[36]:


#Load your nmf_model with the n_components i.e 5
num_topics =  5 
#keep the random_state =40
nmf_model = NMF(n_components=num_topics, random_state=40)
W1 = nmf_model.fit_transform(dtm)
H1 = nmf_model.components_


# In[37]:


#Print the Top15 words for each of the topics
num_words=15
array = np.array(tfidf.get_feature_names_out())
top_words = lambda t: [array[i] for i in np.argsort(t)[:-num_words-1:-1]]
topic_words = ([top_words(t) for t in H1])
topics = [' '.join(t) for t in topic_words]


# In[38]:


array


# In[39]:


topics


# In[40]:


colnames = ["Topic" + str(i) for i in range(nmf_model.n_components)]
docnames = ["Doc" + str(i) for i in range(len(data_clean.Complaint_clean))]
data_doc_topic = pd.DataFrame(np.round(W1, 2), columns=colnames, index=docnames)
significant_topic = np.argmax(data_doc_topic.values, axis=1)
data_doc_topic['dominant_topic'] = significant_topic
data_doc_topic.head()


# In[41]:


data_clean['Topic'] = significant_topic
pd.set_option('display.max_colwidth', -1)
data_clean[['complaint_text','Complaint_clean','category','Topic']][data_clean.Topic==4].head(30)


# In[42]:


temp =data_clean[['complaint_text','Complaint_clean','category','Topic']].groupby('Topic').head(10)
temp.sort_values('Topic')


# In[43]:


#Create the dictionary of Topic names and Topics
topic_mapping = {
    0: 'Bank Account services',
    1: 'Credit card or prepaid card',
    2: 'Others',
    3: 'Theft/Dispute Reporting',
    4: 'Mortgage/Loan'
}

#Replace Topics with Topic Names
data_clean['Topic'] = data_clean['Topic'].map(topic_mapping)
data_clean.head()


# In[44]:


plt.figure(figsize=(12,6))
sns.countplot(x='Topic',data=data_clean)


# In[45]:


training_data = data_clean[['complaint_text','Topic']]


# In[46]:


training_data.head()


# In[47]:


reverse_topic_mapping = {
    'Bank Account services' :0,
    'Credit card or prepaid card':1,
    'Others':2,
    'Theft/Dispute Reporting':3,
    'Mortgage/Loan':4
}
#Replace Topics with Topic Names
training_data['Topic'] = training_data['Topic'].map(reverse_topic_mapping)
training_data.head()


# In[48]:


training_data[['complaint_text','Topic']][training_data.Topic==2].head(30)


# In[49]:


# x - y  split
X = training_data.complaint_text
y = training_data.Topic
# Fit transform the X
count_vect = CountVectorizer()
X_vect = count_vect.fit_transform(X)


# In[50]:


from sklearn.feature_extraction.text import TfidfTransformer
#Write your code here to transform the word vector to tf-idf
#Fit transform word vector to TF-IDF
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_vect)


# In[51]:


from sklearn.model_selection import train_test_split
# train test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.25, random_state=40, stratify=y)


# In[52]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# In[53]:


def display_classification_report(model,metric):
    y_train_pred_proba = model.predict_proba(X_train)
    y_test_pred_proba = model.predict_proba(X_test)
    roc_auc_score_train = round(roc_auc_score(y_train, y_train_pred_proba,average='weighted',multi_class='ovr'),2)
    roc_auc_score_test = round(roc_auc_score(y_test, y_test_pred_proba,average='weighted',multi_class='ovr'),2)
    print("ROC AUC Score Train:", roc_auc_score_train)
    print("ROC AUC Score Test:", roc_auc_score_test)
    metric.append(roc_auc_score_train)
    metric.append(roc_auc_score_test)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    precision_train,recall_train,fscore_train,support_train=precision_recall_fscore_support(y_train,y_train_pred,average='weighted')
    precision_test,recall_test,fscore_test,support_test=precision_recall_fscore_support(y_test,y_test_pred,average='weighted')    
    acc_score_train = round(accuracy_score(y_train,y_train_pred),2)
    acc_score_test = round(accuracy_score(y_test,y_test_pred),2)  
    metric.append(acc_score_train)
    metric.append(acc_score_test)
    metric.append(round(precision_train,2))
    metric.append(round(precision_test,2))
    metric.append(round(recall_train,2))
    metric.append(round(recall_test,2))
    metric.append(round(fscore_train,2))
    metric.append(round(fscore_test,2)) 
    print('Train Accuracy :',acc_score_train)
    print('Test Accuracy :',acc_score_test)    
    model_report_train = classification_report(y_train,y_train_pred)
    model_report_test = classification_report(y_test,y_test_pred) 
    print('Classification Report for Train:\n',model_report_train)
    print('Classification Report for Test:\n',model_report_test)
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(12, 8))
    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    cmp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    cmp.plot(ax=ax)
    plt.xticks(rotation=80)
    plt.show();


# In[54]:


folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 40)
# utility method to get the GridSearchCV object
def grid_search(model,folds,params,scoring):   
    grid_search = GridSearchCV(model,
                                cv=folds, 
                                param_grid=params, 
                                scoring=scoring, 
                                n_jobs=-1, verbose=1)
    return grid_search


# In[55]:


def print_best_score_params(model):
    print("Best Score: ", model.best_score_)
    print("Best Hyperparameters: ", model.best_params_)


# In[56]:


# create MNB model object
mnb = MultinomialNB()


# In[57]:


# fit model
mnb.fit(X_train, y_train)


# In[58]:


# display classification report
metric1=[]
display_classification_report(mnb,metric1)


# In[59]:


# Logistic Regression Classification
log_reg = LogisticRegression(random_state=40,solver='liblinear')
# fit model
log_reg.fit(X_train,y_train)
# display classification report
metric2=[]
display_classification_report(log_reg,metric2)


# Decision Tree Classification

# In[60]:


# Decision Tree Classification
dtc = DecisionTreeClassifier(random_state=40)
# fit model
dtc.fit(X_train,y_train)
# Decision Tree Classification Report
metric3=[]
display_classification_report(dtc,metric3)


# Random Forest Classification

# In[61]:


rf = RandomForestClassifier(n_estimators = 500,random_state=40, n_jobs = -1,oob_score=True)
# fit model
rf.fit(X_train,y_train)

# oob score
print('OOB SCORE :',rf.oob_score_)

# Random Forest Classification Report
metric4=[]
display_classification_report(rf,metric4)


# Multinomial Naive Bayes with GridSearchCV

# In[62]:


mnb = MultinomialNB()

mnb_params = {  
'alpha': (1, 0.1, 0.01, 0.001, 0.0001)  
}

# create gridsearch object
grid_search_mnb = grid_search(mnb, folds, mnb_params, scoring=None)

# fit model
grid_search_mnb.fit(X_train, y_train)

# print best hyperparameters
print_best_score_params(grid_search_mnb)

# Random Forest Classification Report
metric5=[]
display_classification_report(grid_search_mnb,metric5)


# Logistic Regression with GridSearchCV

# In[63]:


# logistic regression
log_reg = LogisticRegression()

# hyperparameter for Logistic Regression
log_params = {'C': [0.01, 1, 10], 
          'penalty': ['l1', 'l2'],
          'solver': ['liblinear','newton-cg','saga']
         }

# create gridsearch object
grid_search_log = grid_search(log_reg, folds, log_params, scoring=None)

# fit model
grid_search_log.fit(X_train, y_train)

# print best hyperparameters
print_best_score_params(grid_search_log)

# Random Forest Classification Report
metric6=[]
display_classification_report(grid_search_log,metric6)


# Decision Tree Classification with GridSearchCV

# In[64]:


dtc = DecisionTreeClassifier(random_state=40)

dtc_params = {
    'max_depth': [5,10,20,30],
    'min_samples_leaf': [5,10,20,30]
}

# create gridsearch object
grid_search_dtc = grid_search(dtc, folds, dtc_params, scoring='roc_auc_ovr')

# fit model
grid_search_dtc.fit(X_train, y_train)

# print best hyperparameters
print_best_score_params(grid_search_dtc)

# Random Forest Classification Report
metric7=[]
display_classification_report(grid_search_dtc,metric7)


# Random Forest Classifier with GridSearchCV

# In[65]:


from sklearn.ensemble import RandomForestClassifier


# In[66]:


rf = RandomForestClassifier(random_state=40, n_jobs = -1,oob_score=True)

# hyperparameters for Random Forest
rf_params = {
    'max_depth': [10,20,30,40],
    'min_samples_leaf': [5,10,15,20,30],
    'n_estimators': [100,200,500,700]
    
}

# create gridsearch object
grid_search_rf = grid_search(rf, folds, rf_params, scoring='roc_auc_ovr')

# fit model
grid_search_rf.fit(X_train, y_train)

# oob score
print('OOB SCORE :',grid_search_rf.best_estimator_.oob_score_)

# print best hyperparameters
print_best_score_params(grid_search_rf)

# Random Forest Classification Report
metric8=[]
display_classification_report(grid_search_rf,metric8)


# In[67]:


table = {'Metric': ['ROC_AUC Score(Train)','ROC_AUC Score(Test)',
                    'Accuracy(Train)','Accuracy(Test)',
                    'Precision(Train)','Precision(Test)',
                    'Recall(Train)','Recall(Test)',
                    'F1-Score(Train)','F1-Score(Test)'
                   ], 
        'Multinomial Naive Bayes': metric1
        }

mnb_metric = pd.DataFrame(table ,columns = ['Metric', 'Multinomial Naive Bayes'] )
log_metric = pd.Series(metric2, name = 'Logistic Regression')
dtc_metric = pd.Series(metric3, name = 'Decision Tree Classifier')
rf_metric = pd.Series(metric4, name = 'Random Forest Classifier')
grid_mnb_metric = pd.Series(metric5, name = 'Multinomial Naive Bayes with GridSearchCV')
grid_log_metric = pd.Series(metric6, name = 'Logistic Regression with GridSearchCV')
grid_dtc_metric = pd.Series(metric7, name = 'Decision Tree Classifier with GridSearchCV')
grid_rf_metric = pd.Series(metric8, name = 'Random Forest Classifier with GridSearchCV')

final_metric = pd.concat([mnb_metric,log_metric,dtc_metric,rf_metric,
                         grid_mnb_metric,grid_log_metric,grid_dtc_metric,grid_rf_metric], axis = 1)

final_metric


# In[68]:


test_complaint= 'I tried to make a transaction at a supermarket retail store, using my chase \
debit/atm card, but the transaction was declined. I am still able to withdraw money out of an \
ATM machine using the same debit card. Please resolve this issue.'


# In[69]:


# vectorize and tf-idf tranform
test = count_vect.transform([test_complaint])
test_tfidf = tfidf_transformer.transform(test)


# In[70]:


# predict
prediction=grid_search_log.predict(test_tfidf)
prediction


# In[71]:


topic_mapping[prediction[0]]


# In[72]:


data


# In[73]:


X_train


# In[75]:


import pickle

pickle.dump(data1,open('data1.pkl','wb'))


# In[ ]:




