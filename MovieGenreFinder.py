import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

data=pd.read_csv('wiki_movie_plots_deduped.csv')
#print(data.head())

movie=data[['Plot','Genre']]
#print(movie.head())

val=movie['Genre'].value_counts()
#print(val)

#val[['drama','comedy','horror','action']].plt.bar()
plt.bar(['drama','comedy','horror','action'],val[['drama','comedy','horror','action']],color='green')
plt.xlabel('Genre')
plt.ylabel('Values')
plt.title('Bar Plot Example')
plt.show()

movies=movie[movie['Genre'].isin(['drama','comedy','horror','action','war','western', 'musical'])]
#print(movies.head())


valCounts=movies['Genre'].value_counts()
labels=['drama','comedy','horror','action','war','western', 'musical']
fig,ax=plt.subplots()
ax.pie(valCounts,labels=labels)
ax.axis('equal')
plt.show()


movies.reset_index(inplace=True,drop=True)
#print(movies)
#print(movies.head())

def Lcase(text):
    return text.lower()
movies['Plot']=movies['Plot'].apply(Lcase)
#print(movies.head(10))

def cleanText(text):
    #remove brackets
    text=re.sub(r'\[[^]]*\]','',text)
    #remove special characters
    text=re.sub(r'[^a-zA-Z\s]','',text)
    #splitting into words
    words=word_tokenize(text)
    stop_words=set(stopwords.words('english'))
    #removing all stop words
    text=[word for word in words if word.lower() not in stop_words]
    return text
movies['resultant_Plot']=movies['Plot'].apply(cleanText)
print(movies.head(10))

df_encoded=pd.get_dummies(movies.Genre)
#print(df_encoded)
movies=pd.concat([movies,df_encoded],axis=1)
#print(movies.head(10))
movies.drop(['Genre','Plot'],inplace=True,axis=1)
print(movies.head(10))