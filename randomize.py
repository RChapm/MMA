#this file will be used to randomize the historical data set which includes the results from compiling win/loss history with streaks and types of wins included

import pandas as pd
import random

#first we will import the dataset
filepath = "/Users/Ryan/Desktop/Apps/AFreshStart/historical.csv"
df = pd.read_csv(filepath, encoding = "ISO-8859-1")

def swap(rownuma,rownumb,colnamea,colnameb):
  f = row[rownuma]
  df.set_value(row[0],colnamea, row[rownumb])
  df.set_value(row[0],colnameb, f)

for row in df.itertuples():
  flip = random.randint(0, 1)
  if (flip == 1):
    swap(5,6,'f1name','f2name')
    swap(7,8,'f1result','f2result')
    swap(9,10,'f1fid','f2fid')
    swap(14,19,'f1fights','f2fights')
    swap(15,20,'f1w','f2w')
    swap(16,21,'f1l','f2l')
    swap(17,22,'f1fws','f2fws')
    swap(18,23,'f1ls','f2ls')
    swap(24,30,'f1SubW','f2SubW')
    swap(25,31,'f1KOW','f2KOW')
    swap(26,32,'f1DecW','f2DecW')
    swap(27,33,'f1SubL','f2SubL')
    swap(28,34,'f1KOL','f2KOL')
    swap(29,35,'f1DecL','f2DecL')

df.to_csv('randhist.csv')
