# https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

# load the dataset
data = open('data/corpus').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])
    texts.append(content[1:])

# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['text'] = ["He told me he didnt understand that I was uncomfortable, despite my obvious attempts of leaving. I didn’t want to lose our mutual friends. I wanted to believe he did nothing wrong. #WhyIDidntReport", "#WhyIDidntReport In july 2015 Somerset Bridgewater hotel Wear i was staying w/fam, When two females lore me to a room one of them passed out and the other one want sexual intercourse. It’s too hard to say but I was raped by forced until I grab all my clothes and left the room."]
trainDF['label'] = [1, 0]
