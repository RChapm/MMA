#This file is intended to iterate through the list of UFC fights in order to track UFC win/loss history in terms of submission vs knockout etc

#we first need to import relevant libraries including pandas numpy etc
import pandas as pd


#we then need to import the dataframe we will use
filepath = "/Users/Ryan/Desktop/Apps/AFreshStart/SHERDOGfights.csv"
df = pd.read_csv(filepath, encoding = "ISO-8859-1")

#we first need to properly sort the file by event ID and main event id in order to track fights in order
df.sort_values(by=['eid', 'mid'])

df['f1SubW'] = 0
df['f1KOW'] = 0
df['f1DecW'] = 0
df['f1SubL'] = 0
df['f1KOL'] = 0
df['f1DecL'] = 0
df['f2SubW'] = 0
df['f2KOW'] = 0
df['f2DecW'] = 0
df['f2SubL'] = 0
df['f2KOL'] = 0
df['f2DecL'] = 0

#the first layer of code will be to iterate through each possible fight ID (upon first inspection the max fight ID is 172941)
for n in range(172941):
  SubW = 0
  KOW = 0
  DecW = 0
  SubL = 0
  KOL = 0
  DecL = 0

  if (n==20):
    print(df.head())
  if (n == 1000) | (n==5000) |(n==10000) |(n==50000)|(n==100000)|(n==150000):
    print(n)
  #the next layer of code needs to iterate through every single fight given each fighter ID
  for row in df.itertuples():
    #print(row[9])
    #index 9 is f1fid
    if (n == row[9]):
      df.set_value(row[0], 'f1SubW',SubW)
      df.set_value(row[0], 'f1KOW',KOW)
      df.set_value(row[0], 'f1DecW',DecW)
      df.set_value(row[0], 'f1SubL',SubL)
      df.set_value(row[0], 'f1KOL',KOL)
      df.set_value(row[0], 'f1DecL',DecL)
      if (row[11] == 'Submission'):
        SubW += 1
      elif (row[11] == 'TKO')|(row[11] == 'KO'):
        KOW += 1
      elif (row[11] == 'Decision'):
        DecW += 1
    elif (n == row[10]):
      df.set_value(row[0], 'f2SubW',SubW)
      df.set_value(row[0], 'f2KOW',KOW)
      df.set_value(row[0], 'f2DecW',DecW)
      df.set_value(row[0], 'f2SubL',SubL)
      df.set_value(row[0], 'f2KOL',KOL)
      df.set_value(row[0], 'f2DecL',DecL)
      if (row[11] == 'Submission'):
        SubL += 1
      elif (row[11] == 'TKO')|(row[11] == 'KO'):
        KOL += 1
      elif (row[11] == 'Decision'):
        DecL += 1

df.to_csv('winhistory.csv')
