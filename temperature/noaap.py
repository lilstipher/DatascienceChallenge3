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
with open('rennes_max_temperature.csv', 'w') as a:
    file = csv.writer(a)
    for key, value in tmax_2004_2017.items():
       file.writerow([key, value])

#there are some NaN value in 2017






