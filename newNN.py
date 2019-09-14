#%matplotlib notebook

#This is a second neural network to predict UFC fights
#This network is different than the previous one because I was able to use a different dataset for which I calculated a variety of metrics of each fighter's history, including wins, losses, streaks, and types of wins and losses for their tenure in the UFC
#Additionally, the data for this network goes from 1993 to 2016 whereas the other network had data from 2013 to 2018.


########IMPORTING LIBRARIES AND DATA########

#Importing relevant libraries
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from subprocess import check_output
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers



# Load dataset and assign to dataframe 'df' using pandas
filepath = "/Users/Ryan/Desktop/Apps/prepped.csv"
df = pd.read_csv(filepath, encoding = "ISO-8859-1")


########CLEANING DATA########

#I will be excluding the first 1000 fights listed because the nature of the UFC is much different now than it was initially in 1993
print("Original: ",df.shape)
df = df[1000:]
print("Subset: ",df.shape)
print(df.head())
df = shuffle(df)
print("Shuffled: ",df.head())


#setting result value for binary y-output
conditions = [
    (df['f1result'] == 'win'),
    (df['f1result'] == 'loss')]
choices = [1, 0]
df['result'] = np.select(conditions, choices)

#dropping null values
df = df.dropna()

#excluding cases where there is no metric data
df[df.Metrictst_2 != 0]
df[df.Metrictst_1 != 0]
print("Dropped values without metric stats: ",df.shape)


########CREATING TRAINING AND TESTING DATASETS########



#These are combined differentials between variables.
#When I check the correlation between each variable and the fight result, if I notice two variables which seem to be not as effective as expected, often the differential is what will provide the predictive information.
df['Age'] = df['f1Age'] - df['f2Age']
df['Reach'] = df['Reach_1'] - df['Reach_2']
df['weight'] = df['weight_1'] - df['weight_2']
df['height'] = df['height_1'] - df['height_2']
df['fightdif'] = df['f1fights'] - df['f2fights']
df['windif'] = df['f1w'] - df['f2w']
df['lossdif'] = df['f1l'] - df['f2l']
df['subdif'] = df['f1SubW'] - df['f2SubW']
df['KOdif'] = df['f1KOW'] - df['f2KOW']
df['Decdif'] = df['f1DecW'] - df['f2DecW']
df['subldif'] = df['f1SubL'] - df['f2SubL']
df['KOldif'] = df['f1KOL'] - df['f2KOL']
df['Decldif'] = df['f1DecL'] - df['f2DecL']
df['subavg'] = df['Sub. Avg_1'] - df['Sub. Avg_2']
df['decexp'] = df['Decdif'] - df['Decldif']
df['subexp'] = df['subdif'] - df['subldif']
df['KOexp'] = df['KOdif'] - df['KOldif']

#from here I export the modified dataset in order to run correlations in R
#I will use this information to determine what variables to include in the neural network
df.to_csv('Rprepped2.csv')

#selecting inputs for the predictive model
#df1 = df[['Age', 'Reach','f1fights', 'f1w', 'f1l', 'f1fws' ,'f1ls','f2fights','f2w', 'f2l', 'f2fws', 'f2ls', 'f1SubW','f1KOW','f1DecW','f1SubL','f1KOL','f1DecL','f2SubW','f2KOW' ,'f2DecW','f2SubL','f2KOL' ,'f2DecL','SLpM_1' , 'Str. Acc_1' , 'SApM_1',  'Str. Def_1',  'TD Avg_1',  'TD Acc_1'  ,'TD Def_1',  'Sub. Avg_1','SLpM_2'  ,'Str. Acc_2'  ,'SApM_2'  ,'Str. Def_2'  ,'TD Avg_2'  ,'TD Acc_2'  ,'TD Def_2'  ,'Sub. Avg_2' ,'f1Age' , 'result']]
df1 = df[['Reach', 'f1Age','f2Age','weight','f1DecL', 'decexp','subexp','KOexp','windif','lossdif', 'f1fws' , 'f2fws','SLpM_1' , 'Str. Acc_1' , 'SApM_1',  'Str. Def_1',  'TD Avg_1', 'TD Acc_1'  ,'TD Def_1', 'SLpM_2'  ,'Str. Acc_2'  ,'SApM_2'  ,'Str. Def_2'  ,'TD Avg_2'  ,'TD Acc_2'  ,'TD Def_2', 'result']]


#making an array
dataset = df1.values

#setting input array(all rows, columns 0-26 (non inclusive))
X = dataset[:,0:26]

#setting output array (all rows, column 26)
Y = dataset[:,26]

#Because data has different types of input ranges, we should scale it from 0-1
#This allows the neural network to determine which parameters are most significant on its own.
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
#now our scaled data is stored as x_scale


#now we should split our data to train and test data set. we will test 30% of data
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)

#we are also going to split the validation and test sets (the function can only do 1 split at a time)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

#Now we have:
#X_train (28 input features, 70% of full dataset)
#X_val (28 input features, 15% of full dataset)
#X_test (28 input features, 15% of full dataset)
#Y_train (1 label, 70% of full dataset)
#Y_val (1 label, 15% of full dataset)
#Y_test (1 label, 15% of full dataset)

#if you want to see the shapes of the arrays use
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

############################ DONE PROCESSING DATA ####################################

#we are going to now set up 1 hidden layer with 40 neurons(ReLu) and 1 ouput(sigmoid)
#we will use sequential model which means we describe the layers in sequence below
model = Sequential([  Dense(40, activation='relu', input_shape=(26,)), Dense(1, activation='sigmoid')])

#Below is an alternative model in which I experimented with regularization and extra hidden layers
#model = Sequential([   Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(28,)),    Dropout(0.3),    Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.01)), Dropout(0.3),    Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.01)),   Dropout(0.3),    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),])


#tell the model which function to optimize with, what loss function, and metrics to track
#sgd is stochastic gradient descent
#loss function for binaries which take value 0,1 is called binary cross entropy
#i will change the data to be 0 and 1 now, but perhaps i can find a better loss function
#and lastly track accuracy
#use either sgd or adam
model.compile(optimizer='sgd',    loss='binary_crossentropy',      metrics=['accuracy'])

#this line of code is what trains the data
#its called fit bc we fit parameters to the data. specify data training on.
#specify size of mini batch and how long we train (epochs). then spec validation data
#this function outputs a history which we save as hist
hist = model.fit(X_train, Y_train,  batch_size=25, epochs=150,  validation_data=(X_val, Y_val))

#evaluate the accuracy on the test set
print('Accuracy of prediction on test set:')
score = model.evaluate(X_test, Y_test)[1]
print(score)
########################### END of network programming ###########################

#next we can model the results if we want using matplotlib

#visualize training and validation loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

#visualize accuracy on training and validation data
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

#the resulting data should typically demonstrate an accuracy score on the test set somewhere between 60 and 67 percent
#This score isn't as good as the other neural network, but I have yet to include fighter background data, which seemed to greatly impact the other model

#If we have an interest in reviewing the weights and biases of the network we can used the following:
#weights = model.layers[0].get_weights()[0]
#biases = model.layers[0].get_weights()[1]
