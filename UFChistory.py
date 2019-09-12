#This file is intended to iterate through the list of UFC fights in order to track UFC fight history and current streaks for each fighter

#we first need to import relevant libraries including pandas numpy etc
import pandas as pd


#we then need to import the dataframe we will use
filepath = "/Users/Ryan/Desktop/Apps/AFreshStart/SHERDOGfights.csv"
df = pd.read_csv(filepath, encoding = "ISO-8859-1")

#we first need to properly sort the file by event ID and main event id in order to track fights in order
df.sort_values(by=['eid', 'mid'])

df['f1fights'] = 0
df['f1w'] = 0
df['f1l'] = 0
df['f1fws'] = 0
df['f1ls'] = 0
df['f2fights'] = 0
df['f2w'] = 0
df['f2l'] = 0
df['f2fws'] = 0
df['f2ls'] = 0

#the first layer of code will be to iterate through each possible fight ID (upon first inspection the max fight ID is 172941)
for n in range(172941):
  uf=0
  uw=0
  ul=0
  uws=0
  uls=0
  if (n==20):
    print(df.head())
  if (n == 1000) | (n==5000) |(n==10000) |(n==50000)|(n==100000)|(n==150000):
    print(n)
  #the next layer of code needs to iterate through every single fight given each fighter ID
  for row in df.itertuples():
    #print(row[9])
    #index 9 is f1fid
    if (n == row[9]):
      df.set_value(row[0], 'f1fights',uf)
      #df[['f1fights']] = uf
      df.set_value(row[0], 'f1w',uw)
      #df[['f1w']] = uw
      df.set_value(row[0], 'f1l',ul)
      #df[['f1l']]=ul
      df.set_value(row[0], 'f1fws',uws)
      #df[['f1ws']] = uws
      df.set_value(row[0], 'f1ls',uls)
      #df[['f1ls']] = uls
      uf +=1
      if (row[7] == 'win'):
        uw += 1
        uws += 1
        uls = 0
      elif (row[7] == 'loss'):
        ul += 1
        uls += 1
        uws = 0
      else:
        uws = 0
        uls = 0
    elif (n == row[10]):
      df.set_value(row[0], 'f2fights',uf)
      #df[['f2fights']] = uf
      df.set_value((row[0]), 'f2w',uw)
      #df[['f2w']] = uw
      df.set_value((row[0]), 'f2l',ul)
      #df[['f2l']]=ul
      df.set_value((row[0]), 'f2fws',uws)
      #df[['f2ws']] = uws
      df.set_value(row[0], 'f2ls',uls)
      #df[['f2ls']] = uls
      uf +=1
      if (row[8] == 'win'):
        uw += 1
        uws += 1
        uls = 0
      elif (row[8] == 'loss'):
        ul += 1
        uls += 1
        uws = 0
      else:
        uws = 0
        uls = 0

df.to_csv('fighthistory.csv')
