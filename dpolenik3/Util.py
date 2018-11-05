import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import numpy as np
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adagrad
from nolearn.lasagne import NeuralNet
from sklearn.metrics import accuracy_score
import time
def loadBeer(filePath='../data/recipeData.csv'):
	recipes_raw = pd.read_csv(filePath,index_col='BeerID',encoding='latin1')
	recipes_train = recipes_raw[['OG','FG','ABV','IBU','Color','BoilSize','BoilTime','Efficiency','SugarScale','StyleID']]
	maskHigh = recipes_train.IBU > 120
	column_name = 'IBU'
	recipes_train.loc[maskHigh, column_name] = 120
	maskLow = recipes_train.IBU < 5
	recipes_train.loc[maskLow, column_name] = 5
	maskHigh = recipes_train.OG > 1.25
	column_name = 'OG'
	recipes_train.loc[maskHigh, column_name] = 1.25
	recipes_label = recipes_raw[['Style']]
	recipes_label=recipes_label['Style'].fillna('N/A')
	X_train, X_test, y_train, y_test  = train_test_split(recipes_train, recipes_label, test_size=0.3, random_state=0)
	encoder = LabelEncoder()
	X_train['SugarScale'] = encoder.fit_transform(X_train['SugarScale']).astype(np.int32)
	return  X_train, X_test, y_train, y_test
def loadBeerRaw(filePath='../data/recipeData.csv'):
	recipes_raw = pd.read_csv(filePath,index_col='BeerID',encoding='latin1')
	maskHigh = recipes_raw.IBU > 120
	column_name = 'IBU'
	recipes_raw.loc[maskHigh, column_name] = 120
	maskLow = recipes_raw.IBU < 5
	recipes_raw.loc[maskLow, column_name] = 5
	maskHigh = recipes_raw.OG > 1.25
	column_name = 'OG'
	recipes_raw.loc[maskHigh, column_name] = 1.25
	recipes_label = recipes_raw[['Style']]
	recipes_label=recipes_label['Style'].fillna('N/A')
	X_train, X_test, y_train, y_test  = train_test_split(recipes_raw, recipes_label, test_size=0.3, random_state=0)
	return  X_train, X_test, y_train, y_test
def loadWine(filePath='../data/winequality-red.csv'):
    wine_raw = pd.read_csv(filePath)
    wine_label = wine_raw["quality"]
    wine_train = wine_raw.drop("quality",1)
    X_train, X_test, y_train, y_test = train_test_split(wine_train, wine_label, test_size=0.1,random_state=0)
    return X_train, X_test, y_train, y_test

def bic_curve(X_train,models):
    bicValues= [m.bic(X_train) for m in models]
    minTest = min(bicValues)
    minPos= bicValues.index(minTest)
    plt.annotate('Number of Clusters in Min Value: '+str(minPos), xy=(minPos, minTest), xytext=(minPos, minTest+5),
        arrowprops=dict(facecolor='black', shrink=0.05),
        )
    plt.plot(components,bicValues , label='BIC')
    plt.legend()
def runNNs(X_train, X_test, y_train, y_test,num_classes,num_features,title,epochs,learningRate):
    val_auc_single_train, val_auc_single_test = np.zeros(epochs), np.zeros(epochs)
    val_auc_double_train, val_auc_double_test = np.zeros(epochs), np.zeros(epochs)
    timeTaken = []
    timeTakenRollingAvg =[]
    layers = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('output', DenseLayer)
           ]
    net1 = NeuralNet(layers=layers,
                 input_shape=(None, num_features),
                 dense0_num_units=512, 
                 dropout0_p=0.1,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 update=adagrad,
                 update_learning_rate=learningRate,
                 verbose=0,
                 eval_size=0.0,
                 max_epochs=1)
    
    for i in range(epochs):
        net1.fit(X_train, y_train)
        pred = net1.predict(X_test)
        pred = pred.astype(int)
        val_auc_single_test[i] = accuracy_score(y_test,pred)
        pred = net1.predict(X_train)
        pred = pred.astype(int)
        val_auc_single_train[i] = accuracy_score(y_train,pred) 
    
    layers = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('output', DenseLayer)
           ]
    net2 = NeuralNet(layers=layers,
                 input_shape=(None, num_features),
                 dense0_num_units=512, 
                 dropout0_p=0.1,
                 dense1_num_units=256,
                 dropout1_p=0.1,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 update=adagrad,
                 update_learning_rate=learningRate,
                 eval_size=0.0,
                 verbose=0,
                 max_epochs=1)
    for i in range(epochs):
        net2.fit(X_train, y_train)
        pred = net2.predict(X_test)
        pred = pred.astype(int)
        val_auc_double_test[i] = accuracy_score(y_test,pred)
        pred = net2.predict(X_train)
        pred = pred.astype(int)
        val_auc_double_train[i] = accuracy_score(y_train,pred)
        
    plt.plot(val_auc_single_test, label="Test Set : Single layer")
    plt.plot(val_auc_single_train,label="Train Set : Single layer")
    plt.plot(val_auc_double_train, label="Train Set : Two layer")
    plt.plot(val_auc_double_test, label="Test Set : Two layer")
    plt.legend(loc='lower right')
    plt.xlabel("Epochs (Back Propagation Loop)")
    plt.ylabel("Percent Correct")
    plt.title(title+str(learningRate))
    plt.savefig("../plots/RP/EM/neuralnet/"+title+str(learningRate)+".png")
    plt.show()