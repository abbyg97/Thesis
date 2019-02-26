#Import Library of Gaussian Naive Bayes model
# from sklearn.naive_bayes import BernoulliNB
# https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
from sklearn.naive_bayes import GaussianNB
import numpy as np
import csv


#assigning predictor and target variables

# path = '/Users/abbygarrett/Documents/Thesis/newtopic/labelsWithCategories.csv'
# # data=open(path, "r")
# # data=data.read()
# # data=data.split("\n")
# text = []
#
# with open(path) as f:
#     reader = csv.reader(f)
#     for row in reader:
#         # for data in row:
#         #     text.append(data)
#         text.append(row)
#
# for x in range(0, 500):
#     text[x]=text.split(",")
# print(text)
#
# while '' in text:
#     text.remove('')

# print(data[1])


x= np.array([["He told me he didnt understand that I was uncomfortable, despite my obvious attempts of leaving. I didn’t want to lose our mutual friends. I wanted to believe he did nothing wrong. #WhyIDidntReport"],["#WhyIDidntReport I didn't report the first time because I was 10 but I knew instinctively that I would not be believed in my family. Which came true when I disclosed in my 30's. Not being believed was wors than the assault."], ["#WhyIDidntReport In july 2015 Somerset Bridgewater hotel Wear i was staying w/fam, When two females lore me to a room one of them passed out and the other one want sexual intercourse. It’s too hard to say but I was raped by forced until I grab all my clothes and left the room."]])
y = np.array([1, 1, 1])


#Create a Gaussian Classifier
# model = BernoulliNB()
model = GaussianNB()

# Train the model using the training sets
model.fit(x, y)

#Predict Output
predicted= model.predict(["#WhyIDidntReport cause you pervs just love to hear the stories, you sickos need to make amends but can't because you believe it's okay, well it's not, stop pushing sex like it's all important and fuck yourself."])
print(predicted);
