#This file is intended to iterate through the list of UFC fights in order to track UFC fight history and current streaks for each fighter

#we first need to import relevant libraries including pandas numpy etc
import pandas as pd


#we then need to import the dataframe we will use
filepath = "/Users/Ryan/Desktop/Apps/AFreshStart/SHERDOGfights.csv"
df = pd.read_csv(filepath, encoding = "ISO-8859-1")

#we first need to properly sort the file by event ID and main event id in order to track fights in order
df.sort_values(by=['eid', 'mid'])

#the first layer of code will be to iterate through each possible fight ID (upon first inspection the max fight ID is 172941)
for n in range(172941):
  uf=0
  uw=0
  ul=0
  uws=0
  uls=0

  #the next layer of code needs to iterate through every single fight given each fighter ID
  for row in df.itertuples():
    #print(row[9])
    #index 9 is f1fid
    if (n == row[9]):
      df['f1fights'] = uf
      df['f1w'] = uw
      df['f1l']=ul
      df['f1ws'] = uws
      df['f1ls'] = uls
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
      df['f2fights'] = uf
      df['f2w'] = uw
      df['f2l']=ul
      df['f2ws'] = uws
      df['f2ls'] = uls
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
