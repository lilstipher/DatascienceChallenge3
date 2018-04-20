import pygrib
import requests
#import urllib3
import json
#rom io import BytesIO
#urllib3.disable_warnings()
#file='gfsanl_3_20171001_0000_000.grb2'
link='https://nomads.ncdc.noaa.gov/data/gfsanl/200403/20040302/gfsanl_3_20040302_0000_006.grb'
#http = urllib3.PoolManager()
#file= http.request('GET', link)
#print(file.status)
print('******Retrieving grib data*******')
 
r = requests.get(link)
#e=json.loads(file.data.decode())
#file = requests.get(link)
#r=requests.get(link)
#print(r.encoding)
#print(r.text)

with open('newdata.grib', 'wb') as f:  
    f.write(r.content)

# Retrieve HTTP meta-data
print(r.status_code)  
print(r.headers['content-type'])  
print(r.encoding)  

a=pygrib.open('newdata.grib')
for i in a:
	print(i)
