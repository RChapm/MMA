#this file will be used to randomize the historical data set which includes the results from compiling win/loss history with streaks and types of wins included

import pandas as pd
import random

#first we will import the dataset
filepath = "/Users/Ryan/Desktop/Apps/AFreshStart/historical.csv"
df = pd.read_csv(filepath, encoding = "ISO-8859-1")

def swap(a,b):
  c = row[a]
  row[a] = row[b]
  row[b] = c

for row in df.itertuples():
  flip = random.randint(0, 1)
  if (flip == 1):
    swap(5,6)
    swap(7,8)
    swap(9,10)
    swap(14,19)
    swap(15,20)
    swap(16,21)
    swap(17,22)
    swap(18,23)
    swap(24,30)
    swap(25,31)
    swap(26,32)
    swap(27,33)
    swap(28,34)
    swap(29,35)

df.to_csv('randhist.csv')
