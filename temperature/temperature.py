from pandas import Series #For plot time ourdataset from 2004 to 2016
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA #Library for Modeling our prediction problem
from sklearn.metrics import mean_squared_error #import pour  le calcul de l'erreur quadratique moyenne
from math import sqrt
import numpy

# load dataset
ourdataset = Series.from_csv('rennes_max_temperature.csv', header=0)
# display first few rows
print(ourdataset.head(20))
# line plot of dataset
ourdataset = ourdataset.astype(float)
ourdataset.plot()
pyplot.show()

#some informations About the model ARIMA
# https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
 #prediction funtion
def predictTemp(coef, pastdata):
	prediction = 0.0
	for i in range(1, len(coef)+1):
		prediction += coef[i-1] * pastdata[-i]
	return prediction
#we work on a non-stationry processus so we transform it to have a stationary processus
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)

ourdatasetValue= ourdataset.values
length = len(ourdataset.values) - 30 #prediction for one month
learningData, testData = ourdatasetValue[0:length], ourdatasetValue[length:] 
#We create 2 Dataset, one for training and one for testing
history = [x for x in learningData]# load training Datas

predictions = list()
for t in range(len(testData)):
	model = ARIMA(history, order=(1,1,1)) # Here we must calculate the order with the AIC number but we choose one
	model_fit = model.fit(trend='nc', disp=False)
	ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
	resid = model_fit.resid #residue for follow arima model
	diff = difference(history)
	predict= history[-1] + predictTemp(ar_coef, diff) + predictTemp(ma_coef, resid)
	predictions.append(predict)
	observed = testData[t]
	history.append(observed)
	print('data predicted=%.3f, Data expected=%.3f' % (predict, observed))
meanError = sqrt(mean_squared_error(testData, predictions))
print('Mean Error %.3f' % meanError)
#Mean Error 3.152 

"""Seabold, Skipper, and Josef Perktold. 
“Statsmodels: Econometric and statistical modeling with python.
” Proceedings of the 9th Python in Science Conference. 2010."""
"""We follow this course for modeling ARIMA model : https://machinelearningmastery.com/make-manual-predictions-arima-models-python/
manual prediction model course by  Jason Brownlee """