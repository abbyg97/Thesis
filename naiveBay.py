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

# nums = [250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
nums = [7000]
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

data = open('./files/7000.tsv').read()
reasons0, tweets0, reasons1, tweets1 = [], [], [], []
for i, line in enumerate(data.split("\n")):
    if(i == 7602):
        continue
    content = line.split("\t")
    if(content[0] == '1'):
        reasons1.append(content[0])
        tweets1.append(content[1])
    elif(content[0]=='0'):
        reasons0.append(content[0])
        tweets0.append(content[1])

#need to randomize here
for x in nums:

    # create a dataframe using texts and lables
    trainDF1 = pandas.DataFrame()
    trainDF1['tweet'] = tweets1
    trainDF1['reason'] = reasons1

    trainDF0 = pandas.DataFrame()
    trainDF0['tweet'] = tweets0
    trainDF0['reason'] = reasons0
    #
    # trainDF0 = trainDF0.sample(n=int(x/2))
    # trainDF1 = trainDF1.sample(n=int(x/2))

    trainDF = trainDF0.append(trainDF1)
    trainDF = trainDF.sample(n=x)

    # print(trainDF)

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

    # # load the pre-trained word-embedding vectors
    # embeddings_index = {}
    # for i, line in enumerate(open('./files/wiki-news-300d-1M.vec')):
    #     values = line.split()
    #     embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')
    #
    # # create a tokenizer
    # token = text.Tokenizer()
    # token.fit_on_texts(trainDF['tweet'])
    # word_index = token.word_index
    #
    # # convert text to sequence of tokens and pad them to ensure equal length vectors
    # train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=300)
    # valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=300)
    #
    # # create token-embedding mapping
    # embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
    # for word, i in word_index.items():
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None:
    #         embedding_matrix[i] = embedding_vector
    # #
    # # #this is what is causing problems
    # def create_rnn_gru():
    #     # Add an Input Layer
    #     input_layer = layers.Input((70, ))
    #
    #     # Add the word embedding Layer
    #     embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    #     embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
    #
    #     # Add the LSTM Layer -- this is where those differ
    #     lstm_layer = layers.LSTM(100)(embedding_layer)
    #
    #     # Add the output Layers
    #     output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    #     output_layer1 = layers.Dropout(0.25)(output_layer1)
    #     output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)
    #
    #     # Compile the model
    #     model = models.Model(inputs=input_layer, outputs=output_layer2)
    #     model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    #
    #     return model

    def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)

        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)

        if is_neural_net:
            predictions = predictions.argmax(axis=-1)

        # return metrics.accuracy_score(valid_y, predictions)
        return [metrics.precision_score(valid_y, predictions, average='micro'), metrics.accuracy_score(valid_y, predictions)]

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

    classifier = create_rnn_gru()

    for x in range(0,10):
        trained = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
        countVals.append(trained[0])
        countValsAcc.append(trained[1])

        trained = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
        wordsVals.append(trained[0])
        wordsValsAcc.append(trained[1])

        trained = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
        rfcountVals.append(trained[0])
        rfcountValsAcc.append(trained[1])

        trained = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
        rfWordVals.append(trained[0])
        rfWordValsAcc.append(trained[1])

        # accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
        # #print "RNN-GRU, Word Embeddings",  accuracy
        # rnnWordVals.append(accuracy)

    count.append(statistics.mean(countVals))
    words.append(statistics.mean(wordsVals))
    # ngrams.append(statistics.mean(ngramsVals))
    # charNGram.append(statistics.mean(charVals))
    rfCount.append(statistics.mean(rfcountVals))
    rfWord.append(statistics.mean(rfWordVals))
    rnnWord.append(statistics.mean(rnnWordVals))
    countAcc.append(statistics.mean(countValsAcc))
    wordsAcc.append(statistics.mean(wordsValsAcc))
    rfCountAcc.append(statistics.mean(rfcountValsAcc))
    rfWordAcc.append(statistics.mean(rfWordValsAcc))

def graphIt(NB, rf, ylabel, title, file):
    plt.scatter(nums, NB, color='red')
    plt.plot(nums, NB, color='red', label='Naive Bayes')
    plt.scatter(nums, rf, color='blue')
    plt.plot(nums, rf, color='blue', label='Random Forest')
    # plt.scatter(nums, rnnWord, color='green')
    # plt.plot(nums, rnnWord, color='green')
    plt.xlabel("Sample Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(file)

graphIt(count, rfCount, "Precision", "Precision of Using Count TF IDF", 'NBCountPre.png')
graphIt(words, rfWord, "Precision", "Precision of Using Word Level TF IDF", 'RfCountPre.png')
graphIt(countAcc, rfCountAcc, "Accuracy", "Accuracy of Using Word Level TF IDF", 'RfCountPre.png')
graphIt(wordsAcc, rfWordAcc, "Accuracy", "Accuracy of Using Word Level TF IDF", 'RfCountPre.png')


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
