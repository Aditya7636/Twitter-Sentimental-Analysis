#!/usr/bin/env python
# coding: utf-8

# In[31]:


#IMPORTING LIBRARIES
#Import all the libraries to be used in this notebook. I prefer to do this at the initial stage and added more libraries as I went along on the project


# In[29]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install tweepy')
get_ipython().system('pip install textblob')
get_ipython().system('pip install wordcloud')
get_ipython().system('pip install pycountry')
get_ipython().system('pip install syspath')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install pandas ')
get_ipython().system('pip install numpy ')
get_ipython().system('pip install nltk')
get_ipython().system('pip install regex')
get_ipython().system('pip install langdetect')
get_ipython().system('pip install sklearn')


# In[2]:


from textblob import TextBlob
import sys
import tweepy as tw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import csv


# In[32]:


#Tweets Mining
#I used the Tweepy library for Python to scrape tweets. You need a developer account with Twitter to get the keys used below for this task.


# In[33]:


auth = tw.OAuthHandler("35kYFWPLP94tZcP548yxCj2Yp","qWC5t05EzIwefVHrBxlstYmgLFVsI713T41Ef0S3xszNb4DUfA")
auth.set_access_token("992432970211516417-V1VRT5ggemgu9w54EWU4zqv6YFFRBkz","bLYMUsPIL4quvyXRaSV1GQ0Jbx4DRnyivhG9BHnO4T1X7")
api = tw.API(auth, wait_on_rate_limit=True)


# In[34]:


search_words = "Ukraine","Russia"
since = "2022-02-20"
query = tw.Cursor(api.search_tweets, q=search_words).items(100)
tweets = [{"Tweets":tweet.text, "Timestamp":tweet.created_at,"tweet_id":tweet.id,"location":tweet.user.location,"retweet":tweet.retweet_count} for tweet in query]
for tweet in query:
    tweets = tweet.text.replace('RT','')
    if tweets.startswith(' @'):
        position = tweets.index(':')
        tweets = tweets[position+2:]
    if tweets.startswith('@'):
        position = tweets.index(' ')
        tweets = tweets[position+2:]
print(tweets)


# In[35]:


#Combining all Tweets into single Pandas Dataframe


# In[3]:


df=pd.read_csv('Twitter_data2.csv',lineterminator='\n', error_bad_lines=False)


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.columns


# In[36]:


#Data Cleaning


# In[8]:


import nltk
from nltk.corpus import stopwords

# Import textblob
from textblob import Word, TextBlob


# In[9]:


import nltk
nltk.download('stopwords')


# In[10]:


stopwords = nltk.corpus.stopwords.words("english")
nltk.download('stopwords')
nltk.download('wordnet')
stopwords = nltk.corpus.stopwords.words("english")
custom_stopwords = ['RT']
import nltk
nltk.download('omw-1.4')


# In[37]:


#Tweets Processing


# In[11]:


def preprocess_tweets(text, custom_stopwords):
    processed_tweet = text
    processed_tweet.replace('[^\w\s]', '')
    processed_tweet = " ".join(word for word in processed_tweet.split() if word not in stopwords)
    processed_tweet = " ".join(word for word in processed_tweet.split() if word not in custom_stopwords)
    processed_tweet = " ".join(Word(word).lemmatize() for word in processed_tweet.split())
    return(processed_tweet)

df["text"] = [str(x).replace(':',' ') for x in df["text"]]

df.head()


# In[12]:


def preprocess_tweets(text, custom_stopwords):
    processed_tweet = text
    processed_tweet.replace('[^\w\s]', '')
    processed_tweet = " ".join(word for word in processed_tweet.split() if word not in stopwords)
    processed_tweet = " ".join(word for word in processed_tweet.split() if word not in custom_stopwords)
    processed_tweet = " ".join(Word(word).lemmatize() for word in processed_tweet.split())
    return(processed_tweet)

df['processed_tweet'] = df['text'].apply(lambda x: preprocess_tweets(x, custom_stopwords))

df.head()


# In[13]:


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
## create function to get polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity
## create two new column
df['Subjectivity']=df['processed_tweet'].apply(getSubjectivity)
df['Polarity']=df['processed_tweet'].apply(getPolarity)
## show new Dataframe
df


# In[14]:


df.rename(columns = {'processed_tweet':'Tweets'}, inplace = True)


# In[15]:


string.punctuation


# In[16]:


def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text
df['Tweets'] = df['Tweets'].apply(lambda x: remove_punct(x))


# In[17]:


df.head()


# In[19]:


df.drop('text',axis=1,inplace=True)


# In[38]:


#Data Exploration


# In[20]:


sorted_tweets = df[['username', 'Tweets', 'retweetcount', 'tweetid','followers']].sort_values(by = 'followers', ascending = False)


# In[21]:


sorted_tweets.head(10)


# In[22]:


most_retweeted = sorted_tweets.iloc[0]
print(most_retweeted.tweetid)


# In[23]:


def word_cloud(wd_list):
    stopwords = set(STOPWORDS)
    all_words = ' '.join([text for text in wd_list])
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        width=1600,
        height=800,
        random_state=1,
        colormap='viridis',
        max_words=80,
        max_font_size=200).generate(all_words)
    plt.figure(figsize=(12, 10))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation="bilinear");
word_cloud(df['Tweets'])


# In[24]:


## create function to compute positive, negative and neutral analysis
def getAnalysis(score):
    if score<0:
        return 'Negative'
    elif score==0:
        return 'Neutral'
    else:
        return 'Positive'
df['Analysis'] = df['Polarity'].apply(getAnalysis)


# In[25]:


df.head()


# In[26]:


ptweet=df[df.Analysis=='Positive']
ptweet=ptweet['Tweets']
round((ptweet.shape[0]/df.shape[0])*100,1)
## get percentage of negative tweets
ntweets=df[df.Analysis=='Negative']
round((ntweets.shape[0]/df.shape[0])*100,1)
## show value counts
df['Analysis'].value_counts()
## plot visulatisation of count
plt.title('Sentiment Analysis')
plt.xlabel('Sentiments') 
plt.ylabel('Counts')
palette='summer'
df['Analysis'].value_counts().plot(kind='bar')
plt.show()


# In[27]:


df.location.value_counts()[:20].plot.bar()


# In[41]:


get_ipython().system('pip install plotly matplotlib seaborn --quiet')


# In[42]:


import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[44]:


import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
## create function to get polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity
## create two new column
df['Subjectivity']=df['Tweets'].apply(getSubjectivity)
df['Polarity']=df['Tweets'].apply(getPolarity)
## show new Dataframe
df


# In[46]:


## create function to compute positive, negative and neutral analysis
def getAnalysis(score):
    if score<0:
        return 'Negative'
    elif score==0:
        return 'Neutral'
    else:
        return 'Positive'
df['Analysis'] = df['Polarity'].apply(getAnalysis)


# In[47]:


df.head()


# In[49]:


#Formating dataframe into CSV file.


# In[48]:


df.to_csv('TwitterFinal_Dataset')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




