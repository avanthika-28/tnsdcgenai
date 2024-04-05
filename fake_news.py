import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
import nltk
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from nltk.stem.porter import PorterStemmer
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')
print ("The shape of the data is (row, column):"+ str(fake.shape))
print ("The shape of the data is (row, column):"+ str(true.shape))
fake['output']='fake'
true['output']='true'
#Concatenating and dropping for fake news
fake['news']=fake['title']+fake['text']
fake_news=fake.drop(['title', 'text'], axis=1)
#Concatenating and dropping for true news
true['news']=true['title']+true['text']
true_news=true.drop(['title', 'text'], axis=1)
#Rearranging the columns
fake_news = fake_news[['subject', 'date', 'news','output']]
true_news = true_news[['subject', 'date', 'news','output']]
frames = [fake_news, true_news]
news_dataset = pd.concat(frames)
news_dataset
#Creating a copy
clean_news=news_dataset.copy()
def review_cleaning(text):
'''Make text lowercase, remove text in square brackets,remove
links, remove punctuation and remove words containing numbers.'''
text = str(text).lower()
text = re.sub('\[.*?\]', '', text)
text = re.sub('https?://\S+|www\.\S+', '', text)
text = re.sub('<.*?>+', '', text)
text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
text = re.sub('\n', '', text)
text = re.sub('\w*\d\w*', '', text)
return text
clean_news['news']=clean_news['news'].apply(lambda
x:review_cleaning(x))
clean_news.head()
stop = stopwords.words('english')
clean_news['news'] = clean_news['news'].apply(lambda x: ' '.join([word
for word in x.split() if word not in (stop)]))
clean_news.head()
ax=sns.countplot(x="output", data=clean_news)
#Setting labels and font size
ax.set(xlabel='Output', ylabel='Count of fake/true',title='Count of
fake and true news')
ax.xaxis.get_label().set_fontsize(15)
ax.yaxis.get_label().set_fontsize(15)
       text = fake_news["news"]
wordcloud = WordCloud(
width = 3000,
height = 2000,
background_color = 'black',
stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
figsize = (40, 30),
facecolor = 'k',
edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
       text = true_news["news"]
wordcloud = WordCloud(
width = 3000,
height = 2000,
background_color = 'black',
stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
figsize = (40, 30),
facecolor = 'k',
edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
       #Extracting 'reviews' for processing
news_features=clean_news.copy()
news_features=news_features[['news']].reset_index(drop=True)
stop_words = set(stopwords.words("english"))
#Performing stemming on the review dataframe
ps = PorterStemmer()
#splitting and adding the stemmed words except stopwords
corpus = []
for i in range(0, len(news_features)):
news = re.sub('[^a-zA-Z]', ' ', news_features['news'][i])
news= news.lower()
news = news.split()
news = [ps.stem(word) for word in news if not word in stop_words]
news = ' '.join(news)
corpus.append(news)
