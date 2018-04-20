# -*- coding: utf-8 -*-
"""
Created on 10/12/2017

@author: Henri Michel KOUASSi, Alban GOUGOUA, Wilfried KOUADIO
@credits: cbothore

__IMPORTANT__: WE USE SOME LIBRARIES IN THIS CODE YOU MUST DOWNLOAD THEM FIRST 
				WITH : pip3 install "library"


"""

#***************Fist part draw zombie walk*******************************
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import networkx as nx
from skimage.graph import route_through_array

input_filename="population-density-map.bmp"
im = Image.open(input_filename)

width=im.size[0]
heigth=im.size[1]

grayim = im.convert("L") #convert to grayscale
colors = grayim.getcolors(width*heigth)
# With the image im, let's generate a numpy array to manipulate pixels
p = np.array(grayim) 
# plot the histogram. We still have a lot of dark colors. Just to check ;-)
#plt.hist(p.ravel())
density = p/255.0
#plot point
plt.imshow(density)
plt.plot(4426,2108,'r*')
plt.plot(669,1306,'r*')
plt.axis('off')
plt.show() #show the map with Brest and Turkey

density = p/255.0 # density Array all 0<density[y,x]<1

def zombies_walk(arrayofweight,startpoint,endpoint):
	arrayofweight=1./(arrayofweight) #because we want max of weight
	zombies_route=route_through_array(arrayofweight, startpoint,endpoint,fully_connected=True,geometric=True)
	#  
	""" we use skimage.graph.MCP class

	A class for finding the minimum weighted path through a given n-d weight array. (that's why
 	we normalize arrayofweight=1./arrayofweight to have most weighted path)
	it returns a List of n-d index tuples defining the path from start to end and the sum of weight
	see more on http://scikit-image.org/docs/dev/api/skimage.graph.html#skimage.graph.route_through_array
	."""
	return zombies_route #return list of pixel route

def draw_walk(imageFile,zombiesWalkList,rgbcolor):
	img= Image.open(imageFile)
	pixelArray = img.load() #create pixel Map
	for i in zombiesWalkList[0]:
		pixelArray[i[1],i[0]] = rgbcolor # change pixel color
	img.show()

a=zombies_walk(density,[1306,669],[2108,4426])
	
draw_walk("population-density-map.bmp",a,(255,0,0,255))

#**********************************time to arrive to brest*******************************************
def duree(array):
   temps=0
   pixelarray=array
   for pixel in pixelarray[0]:
       density_1pixel=density[pixel[0],pixel[1]]          #on recupere ici la densité de chaque pixel qui se trouve sur le chemin
       vitesse_1pixel =((23/24)*density_1pixel+ (1/24))   # on calcule ici la vitesse mise par les zombies pour atteindre chaque pixel en fonction de leur densité
       temps_1pixel=1/vitesse_1pixel                      #on calcule ici le temps mis par les zombies pour atteindre chaque pixel en fonction de leur vitesse 
       temps=temps+temps_1pixel                          # on donne ici le temps total mis par les zombies(en heure) pour atteindre brest qui
   return temps                                                #est la somme de tous les temps mis par les zombies pour parcourir chaque pixel se trouvant sur leur chemin 


temps=duree(a)
temps_jours=temps/24
temps_mois=temps_jours/30
print('duree totale(heure) pour arriver  a brest: %s' % str(temps))  # (5813.35018813) soit 5813h et 35 minutes
print('duree totale(jours) pour arriver  a brest: %s' % str(temps_jours))  #(242.222924505) soit 242 jours 5h et 17 minutes
print('duree totale(mois) pour arriver  a brest: %s' %  str(temps_mois)) #(8.07409748351) soit 8 mois 21 jours

#******************************************2nd part temperature******************************************
#we use here NoAA web api in ulmo library
import ulmo
import pandas
from matplotlib import pyplot
import csv

#https://ulmo.readthedocs.io/en/latest/api.html#module-ulmo.ncdc.ghcn_daily
#data = ulmo.ncdc.ghcn_daily.get_data('GM000010147', as_dataframe=True)
#tm = data['TMAX'].copy()
#tm.value=tm.value/10.0
#tm['value']['1980':'2010'].plot()
#pyplot.show()
#st = ulmo.ncdc.ghcn_daily.get_stations(country='FR', start_year=2004, end_year=2017,as_dataframe=True) use this for retrieve stations infos
#print(st.head()) #show stations Infomartions
#Choose Rennes Station for example ID:FR000007130 

data = ulmo.ncdc.ghcn_daily.get_data('FR000007130', as_dataframe=True)

tmax = data['TMAX'].copy()#maximum temperature
tmax.value=tmax.value/10.0 #convert data to degrees Celsius

tmin = data['TMIN'].copy()# minimum temperature
tmin.value=tmin.value/10.0#convert data to degrees Celsius

tmax_2004_2017=tmax['value']['2004':'2017']
tmin_2004_2017=tmin['value']['2004':'2017']

#tmax_2004_2017.plot()
#tmin_2004_2017.plot()
#pyplot.show()
#tmin_2004_2017=tmin_2004_2017.to_dict('records')
with open('rennes_max_temperature.csv', 'w') as a: #we write data in csv file for easy use with pandas
    file = csv.writer(a)
    for key, value in tmax_2004_2017.items():
       file.writerow([key, value])

#there are some NaN value in 2017 so we will delete 2017 datas
#we will test our model with max temperature

#*************************** temperature prediction""""""""""""""""""""""""""""""""""""""""

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




