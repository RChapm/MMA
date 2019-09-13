#%matplotlib notebook

#This is a neural network to predict UFC fights


########IMPORTING LIBRARIES AND DATA########

#Importing relevant libraries
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from subprocess import check_output
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers

# Load dataset and assign to dataframe 'df' using pandas
filepath = "/Users/Ryan/Desktop/Apps/MMAFULLTDS.csv"
df = pd.read_csv(filepath, encoding = "ISO-8859-1")

#print first five entries to check included variables
print(df.head())

########CLEANING DATA########

#Upon first inspection of the data fights with no 'winby' went to decision
#Replacing null winby values with Decision
df["winby"].fillna("DEC", inplace = True)

#Before running correlations in R, I would like to consider the importance of basic differentials
#Converting metric data into differentials between fighters
df['Reach'] = df['Reach_B'] - df['Reach_R']
df['Weight'] = df['B_Weight'] - df['R_Weight']
df['Height'] = df['Height_B'] - df['Height_R']
df['Age'] = df['B_Age'] - df['R_Age']
df['SAtDif'] = df['SApM__B'] - df['SApM__R']
df['SAcDif'] = df['StrAcc__B'] - df['StrAcc__R']
df['SLDif'] = df['SLpM__B'] - df['SLpM__R']
df['SDDif'] = df['StrDef__B'] - df['StrDef__R']
df['TDAcDif'] = df['TDAcc__B'] - df['TDAcc__R']
df['TDAvDif'] = df['TDAvg__B'] - df['TDAvg__R']
df['TDDDif'] = df['TDDef__B'] - df['TDDef__R']
df['SubDif'] = df['SubAvg__B'] - df['SubAvg__R']

#excluding no contest cases
df[df.winner != 'no contest']
#setting result value for binary y-output
conditions = [
    (df['winner'] == 'red'),
    (df['winner'] == 'blue')]
choices = [0, 1]
df['result'] = np.select(conditions, choices)

#dropping null values
df = df.dropna()

#Using One-hot encoding for the stance variable (converting strings into many dummy variables)
cat_columns = ['Stance_B', 'Stance_R']
df = pd.get_dummies(df, prefix_sep="__", columns=cat_columns)

#its never a bad time to drop null values again
df = df.dropna()


########CREATING TRAINING AND TESTING DATASETS########

#from here I export the modified dataset in order to run correlations in R
#I will use this information to determine what variables to include in the neural network
df.to_csv('Rprepped.csv')


#this first dataframe was the model I used initially. The sheer number of inputs made the neural network fairly unreliable so after running basic correlations in R I settled on the variables in the following dataframe.
#df1 = df[['Age','Reach', 'Height', 'SLpM__B', 'SLpM__R', 'TDAvg__B', 'TDAvg__R', 'StrDef__B', 'StrDef__R', 'SubAvg__B', 'SubAvg__R', 'StrAcc__B', 'StrAcc__R', 'TDAcc__B', 'TDAcc__R', 'Weight', 'Stance_B__Orthodox', 'Stance_B__Southpaw', 'Stance_B__Switch','Stance_R__Orthodox', 'Stance_R__Southpaw', 'Stance_R__Switch', "B_Wrestling", "R_Wrestling", "B_BJJ", 'R_BJJ', 'B_Boxing', 'R_Boxing', 'R_Kickboxing', 'B_Kickboxing','B_MuayTai', 'R_MuayTai', 'B_Freestyle', 'R_Freestyle', 'B_Sambo', 'R_Sambo', 'B_Karate', 'R_Karate', 'B_Judo', 'R_Judo', 'result']]

#selecting Age, Weight, strike/min, takedown avg, and stance for analysis dataframe
df1 = df[['Age', 'B_Age', 'SApM__R', 'TDAvg__B', 'TDAvg__R', 'TDAcc__B', 'TDAcc__R', 'StrAcc__R', 'StrAcc__B', 'B_Freestyle', 'TDDef__R', 'TDDef__B', 'SApM__B', 'StrDef__R', 'SLpM__B', 'SLpM__R', 'StrDef__B', 'R_Freestyle', 'R_MuayTai', 'B_MuayTai', 'result']]
#print(df1.head())


#making an array
dataset = df1.values

#setting input array(all rows, columns 0-20 (non inclusive))
X = dataset[:,0:20]

#setting output array (all rows, column 20)
Y = dataset[:,20]

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

#we are going to now set up 1 hidden layer with 30 neurons(ReLu) and 1 ouput(sigmoid)
#we will use sequential model which means we describe the layers in sequence below
model = Sequential([  Dense(30, activation='relu', input_shape=(20,)), Dense(1, activation='sigmoid')])

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
hist = model.fit(X_train, Y_train,  batch_size=50, epochs=150,  validation_data=(X_val, Y_val))

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

#the resulting data should typically demonstrate an accuracy score on the test set somewhere between 65 and 68 percent
#While in other sports, sport prediction may lead to greater accuracy, in the UFC, the highly volatile nature of the sport leaves to much unpredictabily, making this an acceptable result


#Surprising results from the data (based off of trial and error with the neural network and R output) indicate that fighter Reach doesn't greatly improve the model and a background in muay thai or being a freestyle fighter are by far the greatest predictors of the fight result

#Future goal is to produce data on fight by fight UFC winning streaks, losing streaks, and total UFC experience

#If we have an interest in reviewing the weights and biases of the network we can used the following:
#weights = model.layers[0].get_weights()[0]
#biases = model.layers[0].get_weights()[1]
