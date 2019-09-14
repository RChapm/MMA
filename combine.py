#combine fighter data such as significant strikes, age, stance, background,
import pandas as pd
import numpy as np
from datetime import datetime


#importing the time series file for all the fights
filepath = "/Users/Ryan/Desktop/Apps/randhist.csv"
history = pd.read_csv(filepath, encoding = "ISO-8859-1")

#importing the first fighter profile dataset
filepath = "/Users/Ryan/Desktop/Apps/aFreshStart/name,birth, height, weight, team, origin.csv"
df1 = pd.read_csv(filepath, encoding = "ISO-8859-1")

#importing the second fighter profile dataset
filepath = "/Users/Ryan/Desktop/Apps/aFreshStart/totwins, metrics, Height, weight, stance.csv"
df2 = pd.read_csv(filepath, encoding = "ISO-8859-1")


print(df1.head())
print(df2.head())
print(df1.shape)
print(df2.shape)

#merging fighter profile sets and examining the output (we now have less total fighters but still 1428)
mergedfighter = pd.merge(df1, df2, on='name')

print(mergedfighter.head())
print(mergedfighter.shape)

#the next step is to append each fighter profile to the original dataset for both f1 and f2

merged1 = pd.merge(history, mergedfighter, left_on="f1name", right_on="name")

merged2 = pd.merge(merged1, mergedfighter, left_on="f2name", right_on="name", suffixes=['_1', '_2'])

print(history.head())
print(history.shape)
print(merged1.head())
print(merged1.shape)
print(merged2.head())
print(merged2.shape)

def calculate_age(a,b):
  #since excel stores date values as floats, we can just subtract them from each other
  result = b - a
  return result/365.25
#event_date = row[4]
#birth_date_1 39
#birth_date_2 68
merged2['f1Age'] = np.nan
merged2['f2Age'] = np.nan
for row in merged2.itertuples():
  merged2.set_value(row[0],'f1Age',calculate_age(row[39],row[4]))
  merged2.set_value(row[0],'f2Age',calculate_age(row[68],row[4]))


print(merged2.head(50))
print(merged2.shape)

merged2.to_csv('prepped.csv')
