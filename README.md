# My ANN for MMA Prediction

This repository contains my first project on artificial neural networking. As this has been an iterative process, I have used a variety of data sets and formats for the neural network. As a result, not all of the data sets or programs in the repository are used in the final product. 

# Logistics
To run the code on your own system, be sure to first change the filepaths used to reference the datasets so that it matches the location at which you save them on your own system. This code is run using Python 3 and requires pandas, numpy, matplotlib, sklearn, subprocess, keras, and tensorflow libraries. I recommend running the code via jupyter notebook, and typically prefer formatting the output graphics within the notebook (and hence run %matplotlib notebook before referencing any of the files).

# Latest Iteration
The latest format of the model, which generally outputs a predictive accuracy of 66% is found at newNN.py. This code references the dataset listed prepped.csv. While I have returned accuracies of 70% in some previous iterations, I am sticking with this version of the model as it has fairly high predictive accuracy even without nuanced fighter background information (primarily fighting style). Because this network is based off of a new data set, I have yet to include fighter background information, but that will be my next step in the process.
