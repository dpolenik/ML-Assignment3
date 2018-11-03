import pandas as pd
from sklearn.model_selection import train_test_split

def loadBeer():
	recipes_raw = pd.read_csv('./data/recipeData.csv',index_col='BeerID',encoding='latin1')
	recipes_train = recipes_raw[['OG','FG','ABV','IBU','Color']]
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
	return  X_train, X_test, y_train, y_test
def loadBeerRaw():
	recipes_raw = pd.read_csv('./data/recipeData.csv',index_col='BeerID',encoding='latin1')
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
def loadWine():
    wine_raw = pd.read_csv('./data/winequality-red.csv')
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