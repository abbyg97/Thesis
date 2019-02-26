#Erin Narwhal, 2/23/18, Challenge 2
from __future__ import division
import numpy as np
import matplotlib.pyplot as mplot
import collections
import os
import re
import math
import pandas as pd

#creates path to where the scripts are
path = '/Users/abbygarrett/Documents/Thesis/newtopic/WhyIDidntReport/'

tweets = dict()

tweets =	{
  "username": [],
  "date": [],
  "retweets": [],
  "favorites":[],
  "text":[],
  "geo":[],
  "mentions":[],
  "hashtags":[],
  "id":[],
  "permalink":[],
  "city":[],
  "state":[]
}

username = []
date = []
retweets = []
favorites = []
text = []
geo = []
mentions = []
hashtags = []
id = []
permalink = []
city = []
state = []

patterns = ["because", "I was", "I thought", "I felt", "I didn't"]

#aquired through http://bogdan.org.ua/2007/08/12/python-iterate-and-read-all-files-in-a-directory-folder.html
#loops through all files in the folder, so that it does not have to be hard coded to open each file
for folder in os.listdir(path):
    #skips opening this file
    if(folder == '.DS_Store'):
        continue

    #path to each file (changes when in new loop)
    newpath = path+folder

    for filename in os.listdir(newpath):
        if(filename == '.DS_Store'):
            continue

        #path to each file (changes when in new loop)
        fullpath = newpath+'/'+filename

        words = filename.split("_")
        cityval = words[1]
        stateval=words[2]

        data=open(fullpath, "r")
        data=data.read()
        data=data.split("\n")

        #splits the lexicon text file
        for x in range(0, len(data)-1):
            data[x]=data[x].split(";")

        for x in range(1, len(data) - 1):
            if data[x][8] in id:
                continue
            elif any(z in data[x][4] for z in patterns):
                username.append(data[x][0])
                date.append(data[x][1])
                retweets.append(data[x][2])
                favorites.append(data[x][3])
                text.append(data[x][4])
                geo.append(data[x][5])
                mentions.append(data[x][6])
                hashtags.append(data[x][7])
                id.append(data[x][8])
                permalink.append(data[x][9])
                city.append(cityval)
                state.append(stateval)
            else:
                continue


        tweets["username"] = username
        tweets["date"] = date
        tweets["retweets"] = retweets
        tweets["favorites"] = favorites
        tweets["text"] = text
        tweets["geo"] = geo
        tweets["mentions"] = mentions
        tweets["hashtags"] = hashtags
        tweets["id"] = id
        tweets["permalink"] = permalink
        tweets["city"] = city
        tweets["state"] = state

df = pd.DataFrame(tweets, columns=["username", "date", "retweets", "favorites", "text", "geo", "mentions", "hashtags", "id", "permalink", "city", "state"])
# df_percent = df.sample(frac=0.06)
# df_percent.to_csv('tweets3.csv', sep=',')
df.to_csv('allReasons.csv', sep=',')
