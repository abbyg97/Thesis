# https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import matplotlib.pyplot as plt
import numpy as np
import statistics

nums = [250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
# nums = [7000]
count = []
words = []
ngrams = []
charNGram = []
rfCount = []
rfWord = []
rnnWord = []
countAcc = []
wordsAcc = []
ngramsAcc = []
charNGramAcc = []
rfCountAcc = []
rfWordAcc = []
rnnWordAcc = []

data = open('prepared.csv').read()
reasons, tweets = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split("\t")
    if(i == 7772):
        continue
    reasons.append(content[1])
    tweets.append(content[2])

for x in nums:

    # create a dataframe using texts and lables
    trainDF = pandas.DataFrame()
    trainDF['tweet'] = tweets
    trainDF['reason'] = reasons

    trainDF = trainDF.sample(n=x)

    # split the dataset into training and validation datasets
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['tweet'], trainDF['reason'])

    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    #fit is just doing the count
    count_vect.fit(trainDF['tweet'])

    # transform the training and validation data using count vectorizer object
    #transform learning on other data -- doesnt count words that were not in the fit
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)

    # word level tf-idf
    # basically how often the word appears proprtionally compared to total terms in the document
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=7000)
    tfidf_vect.fit(trainDF['tweet'])
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)

    # ngram level tf-idf
    # same measure as above except ngrams are the combination of n terms together -- probably only concerned with words
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3), max_features=5000)
    tfidf_vect_ngram.fit(trainDF['tweet'])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

    # characters level tf-idf
    # charcter level ngrams -- probably only concerned wth words
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(trainDF['tweet'])
    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

    def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)

        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)

        if is_neural_net:
            predictions = predictions.argmax(axis=-1)

        # return metrics.accuracy_score(valid_y, predictions)
        return metrics.precision_score(valid_y, predictions, average='micro')

    def train_modelAccuracy(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)

        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)

        if is_neural_net:
            predictions = predictions.argmax(axis=-1)

        return metrics.accuracy_score(valid_y, predictions)

    # Naive Bayes on Count Vectors
    #accuracy variable could represent precision depending on the above function call
    #did this as a result of ease of changing things
    countVals = []
    wordsVals = []
    ngramsVals = []
    charVals = []
    rfcountVals = []
    rfWordVals = []
    rnnWordVals = []
    countValsAcc = []
    wordsValsAcc = []
    ngramsValsAcc = []
    charValsAcc = []
    rfcountValsAcc = []
    rfWordValsAcc = []
    rnnWordValsAcc = []

    # classifier = create_rnn_gru()

    for x in range(0,10):
        precision = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
        countVals.append(precision)

        # Naive Bayes on Word Level TF IDF Vectors
        precision = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
        wordsVals.append(precision)

        # Naive Bayes on Ngram Level TF IDF Vectors
        # precision = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
        # ngramsVals.append(precision)
        #
        # # Naive Bayes on Character Level TF IDF Vectors
        # precision = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
        # charVals.append(precision)

        # RF on Count Vectors
        precision = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
        rfcountVals.append(precision)

        # RF on Word Level TF IDF Vectors
        precision = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
        rfWordVals.append(precision)

        #accuracy
        accuracy = train_modelAccuracy(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
        countValsAcc.append(accuracy)

        # Naive Bayes on Word Level TF IDF Vectors
        accuracy = train_modelAccuracy(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
        wordsValsAcc.append(accuracy)

        # RF on Count Vectors
        accuracy = train_modelAccuracy(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
        rfcountValsAcc.append(accuracy)

        # RF on Word Level TF IDF Vectors
        accuracy = train_modelAccuracy(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
        rfWordValsAcc.append(accuracy)

        # accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
        # #print "RNN-GRU, Word Embeddings",  accuracy
        # rnnWordVals.append(accuracy)

    count.append(statistics.mean(countVals))
    words.append(statistics.mean(wordsVals))
    # ngrams.append(statistics.mean(ngramsVals))
    # charNGram.append(statistics.mean(charVals))
    rfCount.append(statistics.mean(rfcountVals))
    rfWord.append(statistics.mean(rfWordVals))
    # rnnWord.append(statistics.mean(rnnWordVals))
    countAcc.append(statistics.mean(countValsAcc))
    wordsAcc.append(statistics.mean(wordsValsAcc))
    rfCountAcc.append(statistics.mean(rfcountValsAcc))
    rfWordAcc.append(statistics.mean(rfWordValsAcc))

# plt.plot(nums, count, 'ro', nums, words, 'bo', nums, ngrams, 'go', nums, charNGram, 'yo')
# nums = [x*0.6 for x in nums]
plt.scatter(nums, count, color='red')
plt.plot(nums, count, color='red', label='Naive Bayes')
plt.scatter(nums, rfCount, color='blue')
plt.plot(nums, rfCount, color='blue', label='Random Forest')
# plt.scatter(nums, rnnWord, color='green')
# plt.plot(nums, rnnWord, color='green')
plt.xlabel("Sample Size")
plt.ylabel("Precision")
plt.title("Precision of Using Count TF IDF")
plt.legend(loc='lower right')
plt.show()

plt.scatter(nums, words, color='red')
plt.plot(nums, words, color='red', label='Naive Bayes')
plt.scatter(nums, rfWord, color='blue')
plt.plot(nums, rfWord, color='blue', label='Random Forest')
plt.xlabel("Sample Size")
plt.ylabel("Precision")
plt.title("Precision of Using Word Level TF IDF")
plt.legend(loc='lower right')
plt.show()

#accuracy graphs
plt.scatter(nums, countAcc, color='red')
plt.plot(nums, countAcc, color='red', label='Naive Bayes')
plt.scatter(nums, rfCountAcc, color='blue')
plt.plot(nums, rfCountAcc, color='blue', label='Random Forest')
plt.xlabel("Sample Size")
plt.ylabel("Accuracy")
plt.title("Accuracy of Using Count TF IDF")
plt.legend(loc='lower right')
plt.show()

plt.scatter(nums, words, color='red')
plt.plot(nums, words, color='red', label='Naive Bayes')
plt.scatter(nums, rfWordAcc, color='blue')
plt.plot(nums, rfWordAcc, color='blue', label='Random Forest')
plt.xlabel("Sample Size")
plt.ylabel("Accuracy")
plt.title("Accuracy of Using Word Level TF IDF")
plt.legend(loc='lower right')
plt.show()
# plt.savefig('RfWordsAcc.png')

# # reasons = ["1", "2", "3", "4", "5", "6", "7", "8"]
# reasons = ["Shame", "Denial", "Fear", "Hopeless", "Memory", "Lack of Info", "Protect Attacker", "Age"]
# prevelance = [22.59565667, 8.518614271, 24.72854188, 26.70630817, 4.550155119, 10.22492244, 2.998965874, 8.673733195]
#
# index = np.arange(len(reasons))
#
# plt.xticks(index, reasons, fontsize=11)
# plt.xlabel("Reasons")
# plt.ylabel("Percent Appeared")
# plt.title("Reasons People Do Not Report Assault")
# plt.bar(index, prevelance, align="center", color="SpringGreen")
# # plt.legend([("1", "Shame"), ("2", "Denial")])
# plt.show()
