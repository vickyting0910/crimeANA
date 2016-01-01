from __future__ import division #force division to be floating point
import geohash
import psycopg2
import sys, os, glob
import psycopg2.extras
import math
from math import sin, cos, sqrt, atan2, radians
from osgeo import ogr, osr
import osgeo.osr
import datetime, pytz
from datetime import timedelta, datetime, date,time
from dateutil import tz
from pandas import *
import numpy as num
from nltk import *
from osgeo import gdal
import itertools as it, operator as op
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import fiona
import pysal
from shapely.geometry import mapping, LineString, MultiPoint
from pyproj import *

# Set up our R namespaces
R = rpy2.robjects.r #run R in python namespaces
DTW = importr('dtw') #import R packages
	
con = None

#do a coresponding shapefile
driver = ogr.GetDriverByName('ESRI Shapefile')
dataset = driver.Open(r'okcities.shp')
layer = dataset.GetLayer()
outSpatialRef = layer.GetSpatialRef()
#outSpatialRef = osr.SpatialReference()
#outSpatialRef.ImportFromEPSG(3005u

try:
#connect with postgresql
	con = psycopg2.connect(database='*****', user='*****', password = '*****', host='*****', port = *****)
	con.autocommit = True #any change would be effective immediately
	cur = con.cursor()

#get the file list
	#cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'gps_point_y20%' AND table_name NOT LIKE 'gps_point_y20%m%' AND table_name NOT LIKE 'gps_point_y2011%' ORDER BY table_name ASC;")
	#cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'gps_point_y2009m1%p%' ORDER BY table_name ASC")
	cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND (table_name LIKE 'gps_point_y2009m1%p%') OR (table_name = 'gps_point_y2010m01p01') OR (table_name = 'gps_point_y2009m09p08') ORDER BY table_name ASC")
	tablename=cur.fetchall()
	
#resample by certain time intervals
	#get time
	mytable=[]
	for x in tablename:
		for y in x:
			mytable.append(y)
		
	mytime1=[]
	mylong1=[]
	mylat1=[]
	myoff1=[]
	for x in mytable: #SET local timezone to 'America/Chicago'; AT TIME ZONE 'UTC'
		#print y
		cur.execute("SELECT timestamp_utc FROM {mytable} WHERE timestamp_utc >= '2009-10-01 05:00:00' AND timestamp_utc <= '2010-01-01 06:00:00' ORDER BY timestamp_utc ASC; ".format(mytable=x)) 
		# time must be by certain orders
		mytime = cur.fetchall()
		mytime1.append(mytime)
		cur.execute("SELECT longitude FROM {mytable} WHERE timestamp_utc >= '2009-10-01 05:00:00' AND timestamp_utc <= '2010-01-01 06:00:00' ORDER BY timestamp_utc ASC".format(mytable=x)) 
		mylong = cur.fetchall()
		mylong1.append(mylong)
		cur.execute("SELECT latitude FROM {mytable} WHERE timestamp_utc >= '2009-10-01 05:00:00' AND timestamp_utc <= '2010-01-01 06:00:00' ORDER BY timestamp_utc ASC".format(mytable=x)) 
		mylat = cur.fetchall()
		mylat1.append(mylat)
		cur.execute("SELECT demographicid FROM {mytable} WHERE timestamp_utc >= '2009-10-01 05:00:00' AND timestamp_utc <= '2010-01-01 06:00:00' ORDER BY timestamp_utc ASC".format(mytable=x)) 
		myoff = cur.fetchall()
		myoff1.append(myoff)
			
	#whole tables
	mytime2=[]
	mylong2=[]
	mylat2=[]
	myoff2=[]
	for a, b, c, d in zip(mytime1, mylong1, mylat1, myoff1):
		for e, f, g, h in zip(a, b, c, d):
			mytime2.append(e)
			mylong2.append(f)
			mylat2.append(g)
			myoff2.append(h)

	#tuple in list to list
	mytime3=[]
	mylong3=[]
	mylat3=[]
	myoff3=[]
	for x, y, z, w in zip(mytime2, mylong2, mylat2, myoff2):
		for p, q,r,s in zip(x, y, z, w):
			#convert utc to local time
			local = pytz.timezone ("America/Chicago") 
			#Central Time: -06:00(utc offset) -05:00 (utc DST offset)
			utc_dt = p.replace(tzinfo=pytz.utc)
			st_dt = local.normalize(utc_dt.astimezone(local)).replace(tzinfo=None)
			mytime3.append(st_dt)
			mylong3.append(q)
			mylat3.append(r)
			myoff3.append(s)
	
	#group by demographicid
	groups=[]
	uniquekeys=[]
	for i, j in it.groupby(tuple(sorted(myoff2)), key=op.itemgetter(0)):
		groups.append(list(j))
		uniquekeys.append(i)
	
#create base time series at a specific time interval
	#mybase=date_range(datetime(2009,9,25,0,0,0), datetime(2010,1,5,23,59,59), freq='30min')
	mybase=date_range(mytime3[0], mytime3[len(mytime3)-1]+timedelta(weeks=0, days=0, hours=0,minutes=0, seconds=0), freq='30min')
	#date_range(mytime3[0], mytime3[len(mytime3)-1]+timedelta(weeks=0, days=0, hours=23-(mytime3[len(mytime3)-1].hour),minutes=30, seconds=0), freq='30min')
	mybseries = Series(num.zeros(len(mybase)), mybase)

	#how many elements for a day
	arrinterval=(timedelta(days=1).total_seconds())/(timedelta(weeks=0, days=0, hours=0, minutes=30, seconds=0).total_seconds())
	z = (mytime3[len(mytime3)-1]-mytime3[0]).days #total days

	#build time series for longitude and latitude
	serlong = Series (mylong3,mytime3)
	serlat = Series (mylat3,mytime3)

	#add header in order to remove 0 for resample and select specific offenders
	dflong=DataFrame(serlong,columns=['Along'])
	dflat=DataFrame(serlat,columns=['Alat'])
	dflong['mydate']=dflong.index
	dflat['mydate']=dflat.index

	#add demographicid in dataframes
	dflong['longoff']=myoff3
	dflat['latoff']=myoff3

	#sort by date
	dflong1=dflong.sort_index(by=['mydate'], ascending=[True])
	dflat1=dflat.sort_index(by=['mydate'], ascending=[True])
	
	#get rid of 0
	mylong = dflong1[dflong1.Along != 0]
	mylat = dflat1[dflat1.Alat != 0]
	

		
#select specific offenders
	mykey=uniquekeys[200:220]
	
	for t in [157587]:
		mylongs = mylong.loc[mylong['longoff'] == t]
		mylats = mylat.loc[mylat['latoff'] == t]
			
	#average longitude and latitude every certain time interval (5 second as an example)
		long5s = mylongs.resample('30min', how = 'mean')
		lat5s = mylats.resample('30min', how = 'mean')
		#long5s['mydate']=long5s.index
		lat5s['mydate']=lat5s.index		
		
	#convert from dataframe to time series
		longtser = long5s.unstack()['Along']
		lattser = lat5s.unstack()['Alat']
		
	#add base
		Blong5s = longtser + mybseries
		Blat5s = lattser + mybseries
	

		
#pre-processing
	#numpy to list
		long5sl=num.array(Blong5s).reshape(-1,).tolist()
		lat5sl=num.array(Blat5s).reshape(-1,).tolist()
	#replace nan to 0
		listlong = [0 if math.isnan(x) else x for x in long5sl]
		listlat = [0 if math.isnan(x) else x for x in lat5sl]
		
	#1. no data assigned and no previous points, then assign the first record
	#2. no data with in the range(assuming no move), then insert the previous point
		longlong=[]
		latlat=[]
		for r in range(1, int(arrinterval*z), int(arrinterval)):
			#get the index of the nan (0) value from each individual line
			longindex = [i for i, x in enumerate(listlong[r: int(r+arrinterval)]) if x == 0] 
			latindex = [i for i, x in enumerate(listlat[r: int(r+arrinterval)]) if x == 0]
			#currect line
			w=listlong[r: int(r+arrinterval)]
			s=listlat[r: int(r+arrinterval)]
			myran=range(48)
			mydiff=list(set(longindex).symmetric_difference(set(myran)))
			for x, y in zip(longindex, latindex):
				if longindex[0] == 0 and x < mydiff[0] :
					d=mydiff[0]
					w[x]=w[d]
					s[y]=s[d]
				else:
					w[x]=w[x-1]
					s[y]=s[y-1]
			longlong.append(w)
			latlat.append(s)
		
		#del longlong[-1]
		#del latlat[-1]
		longlong1=num.asarray(longlong)
		latlat1=num.asarray(latlat)
		longlo=longlong1.reshape(longlong1.shape[0]*longlong1.shape[1],).tolist()
		latla=latlat1.reshape(latlat1.shape[0]*latlat1.shape[1],).tolist()
	
		"""
		#assign from the previous or the next days
		for r in range(1, int(arrinterval*z), int(arrinterval)): 
			longindex = [i for i, x in enumerate(listlong[r: int(r+arrinterval)]) if x == 0] 
			latindex = [i for i, x in enumeraTypeError: in method 'Geometry_Transform', argument 2 of type 'OSRCoordinateTran
sformationShadow *'te(listlat[r: int(r+arrinterval)]) if x == 0]
			#next line number
			a = listlong[int(r+arrinterval): int(r+arrinterval+arrinterval)]
			b = listlat[int(r+arrinterval): int(r+arrinterval+arrinterval)]
			#the next two lines
			c=listlong[int(r+afor g in range(textana1.shape[0]):
			#textana1[i]
			alignment = R.dtw(textana1[g], textana1[0], keep=True)
			dist = alignment.rx('distance')[0][0]
			nordist = alignment.rx('normalizedDistance')[0][0] 
			myDTW.append(nordist)rrinterval+arrinterval): int(r+arrinterval+arrinterval+arrinterval)]
			d=listlat[int(r+arrinterval+arrinterval): int(r+arrinterval+arrinterval+arrinterval)]
			#current line
			w=listlong[r: int(r+arrinterval)]
			s=listlat[r: int(r+arrinterval)]
			#previous line
			if r !=1:
				p=listlong[int(r-arrinterval): r]
				q=listlat[int(r-arrinterval): r]
			for x,y in zip(longindex, latindex): 
				#index substitution for the first line
				if r == 1:
					w[x]=a[x]
					s[y]=b[y]
				#index substitution for the following using the previous tracks
				if r != 1:
					w[x]=p[x]
					s[y]=q[y]
			longlong.append(w)
			latlat.append(s)
		http://rebuzz.tw/2013/10/how-porn-industry-use-big-data-to-innovative.html
		longlong1=num.array(longlong)
		latlat1=num.array(latlat)
		longlo=longlong1.reshape(len(listlong)-1,).tolist()
		latla=latlat1.reshape(len(listlat)-1,).tolist()
		"""
		
	#combine two lists
	#longlat = map(add, long5sl, lat5sl) #for list of tuple
		longlat = [list(x) for x in zip(longlo, latla)]
		
#convert longitude and latitude to geohash codes
	#geohash codes
		mygeo=[]
		for w in longlat:
			#i[0] longtitude, i[1] latitude
			mygeo.append(geohash.encode(w[1], w[0]))
		
		mygeo7=[]
		for k in range(len(mygeo)):
			mygeo7.append(mygeo[k][0:7])
			
	#change inner tuple as the list for i in longlat: mynew.append(list(i))
		mypoints = [x+[y] for x,y in zip(longlat, mygeo7)] #combine x, y coordinates and geohash codes
		pointbase = Series (mypoints, mybase[1:]) #combine longitude, latitude and geohash codes
		
	#spatial probability mapping
		mydate=date_range(mytime3[0], mytime3[len(mytime3)-1]+timedelta(weeks=0, days=0), freq='1d')
		datediff=[]
		date_start=[]
		date_end=[]
		for i in range(longlong1.shape[0]):
			#a=date(2009,10,1)+timedelta(days=i)
			#b=date(2009,10,2)+timedelta(days=i)
			a=mydate[0]+timedelta(days=i)
			b=mydate[1]+timedelta(days=i)
			c=str(a)+' to '+str(b)			
			datediff.append(c) #daily difference string
			date_start.append(a) #start date datetime
			date_end.append(b) #end date datetime
		
		timeinterval=30
		longslice=[]
		latslice=[]
		for q in range(0, int(arrinterval*timeinterval), timeinterval):
			for p in range(longlong1.shape[0]):
				a=mydate[0]+timedelta(days=p)
				c=(a+timedelta(minutes=q)).to_datetime() #start time
				d=(a+timedelta(minutes=q+timeinterval)).to_datetime() #end_time
				speclong=mylongs[c:d] 
				#slice the date range
				speclong1=speclong.unstack()['Along']
				speclat=mylats[c:d] 
				#slice the date range
				speclat1=speclat.unstack()['Alat']
				longslice.append(speclong1)
				latslice.append(speclat1)
		overlong=[]
		overlat=[]
		for q in range(15, int(arrinterval*timeinterval), timeinterval):
			for p in range(longlong1.shape[0]):
				a=mydate[0]+timedelta(days=p)
				c=(a+timedelta(minutes=q)).to_datetime() #start time
				d=(a+timedelta(minutes=q+timeinterval)).to_datetime() #end_time
				speclong=mylongs[c:d] 
				#slice the date range
				speclong1=speclong.unstack()['Along']
				speclat=mylats[c:d] 
				#slice the date range
				speclat1=speclat.unstack()['Alat']
				overlong.append(speclong1)
				overlat.append(speclat1)
		
		point30=[]
		multipoints=[]
		mycoor=[]
		coordinates=[]
		myha=[]
		multigeo=[]
		for i,j in zip(range(0, int(z*arrinterval),z), range(int(arrinterval))): #zip(longslice, latslice)
			p=longslice[i:i+z] #get one 30 mins slices for three months
			q=latslice[i:i+z]
			for r,s in zip(p,q): #loop in the slice of a single day
				for k, l in zip(r,s): #loop every longitude and latitude
					point30.append(MultiPoint([(k,l)]))
					myha.append(geohash.encode(l,k)[0:8])
					mycoor.append((k,l))

			multipoints.append(point30)
			multigeo.append(myha)
			coordinates.append(mycoor)
			point30=[]
			myha=[]
			mycoor=[]
		
		overpoint=[]
		overpoints=[]
		overha=[]
		overgeo=[]
		for i,j in zip(range(0, int(z*arrinterval),z), range(int(arrinterval))): #zip(longslice, latslice):
			p=longslice[i:i+z] #get one 30 mins slices for three months
			q=latslice[i:i+z]
			for r,s in zip(p,q): #loop in the slice of a single day
				for k, l in zip(r,s): #loop every longitude and latitude
					overpoint.append(MultiPoint([(k,l)]))
					overha.append(geohash.encode(l,k)[0:8])

			overpoints.append(overpoint)
			overgeo.append(overha)
			overpoint=[]
			overha=[]

		"""
		#export point shapefiles every certain time intervals with geohash codes
		for i, j, m in zip(multipoints, range(len(multipoints)), multigeo):
			schema = {'geometry': 'MultiPoint','properties': {'points'+str(j): 'str'}}
			with fiona.open('*****.shp','w','ESRI Shapefile', schema) as e:
				for k, l, n in zip(i, range(len(i)), m):
					e.write({'geometry':mapping(k), 'properties':{'points'+str(j):str(n)}})
		mylist=[] #convert string in list of list
		listgeo=[]
		for i in multigeo:
			for j in i:
				mylist.append(re.sub("[^\w]", " ",  j).split())
			listgeo.append(mylist)
			mylist=[]
		"""

		#create probability
		"""
		#group all similar geohash codes
		groups1=[]
		groupgeo=[]
		uniquekeys1=[]
		unigeo=[]
		for g in listgeo:
			for i,j in it.groupby(sorted(g), key=op.itemgetter(0)):
				groups1.append(list(j))
				uniquekeys1.append(i)
			groupgeo.append(groups1)
			unigeo.append(uniquekeys1)
			groups1=[]
			uniquekeys1=[]
		"""
		myprob=[]
		probability=[]
		for i in multigeo: #all geohash , uniquekey
			for p in i:
				myprob.append(((i.count(p))/len(i))) #list.count(element)
			probability.append(myprob)
			myprob=[]
		"""
		#export point shapefiles every certain time intervals with probabilities
		for i, j, m in zip(multipoints, range(len(multipoints)), probability):
			schema = {'geometry': 'MultiPoint','properties': {'points'+str(j): 'float'}}
			with fiona.open('*****.shp','w','ESRI Shapefile', schema) as e:
				for k, l, n in zip(i, range(len(i)), m):
					e.write({'geometry':mapping(k), 'properties':{'points'+str(j):float(n)}})
		"""
#create geohash polygons
		geohashcode=[0,1,2,3,4,5,6,7,8,9,'b','c','d','e','f','g','h','j','k','m','n','p','q','r','s','t','u','v','w','x','y','z']
		geoodd=num.array([['b','c','f','g','u','v','y','z'],[8,9,'d','e','s','t','w','x'],[2,3,6,7,'k','m','q','r'],[0,1,4,5,'h','j','n','p']])
		geoeven=num.array([['p','r','x','z'],['n','q','w','y'],['j','m','t','v'],['h','k','s','u'],[5,7,'e','g'],[4,6,'d','f'],[1,3,9,'c'],[0,2,8,'b']])
		example="9y6rhsmf"
		listex=list(example)
		rightmin=num.array([90,0]) #first level geohash coordinates
		rightmax=num.array([90,45])
		leftmin=num.array([45,0])
		leftmax=num.array([45,45])

		mygeohash=[]
		for i,j in zip(listex[1:], range(2,len(listex)+1)):
			if j % 2 ==0: #even number
				if j == 2:

					index1=8-num.where(geoeven==i)[0][0]
					index2=num.where(geoeven==i)[1][0]
					gxRmin=num.array([leftmin[0]+num.true_divide((rightmin[0]-leftmin[0]),4)*(index2+1),leftmin[1]+num.true_divide((leftmax[1]-leftmin[1]),8)*(index1-1)])
					gxRmax=num.array([leftmin[0]+num.true_divide((rightmin[0]-leftmin[0]),4)*(index2+1),leftmin[1]+num.true_divide((leftmax[1]-leftmin[1]),8)*index1])
					gxLmin=num.array([leftmin[0]+num.true_divide((rightmin[0]-leftmin[0]),4)*index2,leftmin[1]+num.true_divide((leftmax[1]-leftmin[1]),8)*(index1-1)])
					gxLmax=num.array([leftmin[0]+num.true_divide((rightmin[0]-leftmin[0]),4)*index2,leftmin[1]+num.true_divide((leftmax[1]-leftmin[1]),8)*index1])
				else:
					index1=8-num.where(geoeven==i)[0][0]
					index2=num.where(geoeven==i)[1][0]
					gxRmin, gxRmax, gxLmin, gxLmax = num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),4)*(index2+1),gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),8)*(index1-1)]), num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),4)*(index2+1),gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),8)*index1]), num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),4)*index2,gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),8)*(index1-1)]), num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),4)*index2,gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),8)*index1])
					#gxRmin=num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),4)*(index2+1),gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),8)*(index1-1)])
					#gxRmax=num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),4)*(index2+1),gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),8)*index1])
					#gxLmin=num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),4)*index2,gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),8)*(index1-1)])
					#gxLmax=num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),4)*index2,gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),8)*index1])

				
			elif j %2 == 0: #odd number
				index1=4-num.where(geoodd==i)[0][0]
				index2=num.where(geoodd==i)[1][0]
				#do it at once
				gxRmin, gxRmax, gxLmin, gxLmax = num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),8)*(index2+1),gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),4)*(index1-1)]), num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),8)*(index2+1),gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),4)*index1]), num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),8)*index2,gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),4)*(index1-1)]), num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),8)*index2,gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),4)*index1])
				#gxRmin=num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),8)*(index2+1),gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),4)*(index1-1)])
				#gxRmax=num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),8)*(index2+1),gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),4)*index1])
				#gxLmin=num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),8)*index2,gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),4)*(index1-1)])
				#gxLmax=num.array([gxLmin[0]+num.true_divide((gxRmin[0]-gxLmin[0]),8)*index2,gxLmin[1]+num.true_divide((gxLmax[1]-gxLmin[1]),4)*index1])

			mygeohash.append([gxRmin, gxRmax, gxLmin, gxLmax])

		#select the most points with the same geohash codes and average long & lat
		mostlong=[]
		mostlat=[]
		mlong=[]
		mlat=[]
		for i in range(len(probability)):
			highprob=num.array(probability[i]).max() 
			#select the high prob (the most points with the same geohash code)
			highindex=[s for s, x in enumerate(probability[i]) if x == highprob] 
			#get the indices with the highest probability
			for k in highindex:
				mlong.append(coordinates[i][k][0]) #get longitude
				mlat.append(coordinates[i][k][1]) #get latitude
			
			mostlong.append(num.array(mlong).mean())
			mostlat.append(num.array(mlat).mean())
			mlong=[]
			mlat=[]

		#export as shapefile
		
		averagepoint=[]
		#for i, j, k,l in zip(mostlong, mostlat, mostlong[1:], mostlat[1:]):
		for i, j in zip(mostlong, mostlat):
			#averagepoint.append(LineString([(i,j), (k,l)]))
			averagepoint.append(MultiPoint([(i,j)]))
		"""
		schema = {'geometry': 'MultiPoint','properties': {'points': 'int'}}
		with fiona.open('*****.shp','w','ESRI Shapefile', schema) as e:
			for i in averagepoint:
				e.write({'geometry':mapping(i), 'properties':{'points':int(1)}})
		"""
		
		
		#create prob for overlay minutes
		overprob=[]
		oprobability=[]
		for i in overgeo: #all geohash , uniquekey
			for p in i:
				overprob.append(((i.count(p))/len(i))) #list.count(element)
			oprobability.append(overprob)
			overprob=[]

		for i, j, m in zip(overpoints, range(len(overpoints)), oprobability):
			schema = {'geometry': 'MultiPoint','properties': {'points'+str(j): 'float'}}
			with fiona.open('*****.shp','w','ESRI Shapefile', schema) as e:
				for k, l, n in zip(i, range(len(i)), m):
					e.write({'geometry':mapping(k), 'properties':{'points'+str(j):float(n)}})

#calculate distance and speed and resmaple every specific time
		#calculate individual distances
		"""
		cadistance=[]
		mydate=[]
		for m in range(len(datediff)):
			startlong=mylongs.index.searchsorted(date_start[m]) #get the indiex of the start date
			endlong=mylongs.index.searchsorted(date_end[m]) #get the indiex of the end date
			speclong=mylongs.ix[startlong:endlong].sort_index(by=['mydate'], ascending=[True]) 
			#slice the date range
			speclong1=speclong.unstack()['Along']

			startlat=mylats.index.searchsorted(date_start[m]) #get the indiex of the start date
			endlat=mylats.index.searchsorted(date_end[m]) #get the indiex of the end date
			speclat=mylats.ix[startlat:endlat].sort_index(by=['mydate'], ascending=[True]) 
			#slice the date range
			speclat1=speclat.unstack()['Alat']
			#mylongs.resample('30min', how = 'mean')
			Rad = 6371.0
			mydate.append(speclong1.index)
			for p, q, r,s in zip(speclong1, speclat1, speclong1[1:], speclat1[1:]):
				lat1 = radians(q)
				lon1 = radians(p)
				lat2 = radians(s)
				lon2 = radians(r)
				dlon = lon2 - lon1
				dlat = lat2 - lat1
				a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
				c = 2 * atan2(sqrt(a), sqrt(1-a))
				cadistance.append(Rad * c)
		"""
		"""
	#calculate interpolation distance
		#time slice #No records between 16:56:32 and 18:54:29 in 12/25/2009 => 0
		longslice1=[]
		latslice1=[]
		for p in range(longlong1.shape[0]):
			for q in range(0, int(arrinterval*timeinterval), timeinterval):
				a=mydate[0]+timedelta(days=p)
				c=(a+timedelta(minutes=q)).to_datetime() #start time
				d=(a+timedelta(minutes=q+timeinterval)).to_datetime() #end_time
				speclong=mylongs[c:d] 
				#slice the date range
				speclong1=speclong.unstack()['Along']
				speclat=mylats[c:d] 
				#slice the date range
				speclat1=speclat.unstack()['Alat']
				longslice1.append(speclong1)
				latslice1.append(speclat1)
		#interpolation
		interlong=[]
		interlat=[]
		for i,j in zip(range(len(longslice1)-1), range(1,len(longslice1))):
			if longslice1[j].empty:
				interlong.append(0)
				interlat.append(0)
			elif longslice1[i].empty:
				interlong.append(0)
				interlat.append(0)
			else:
				a=longslice1[i][-1]
				b=longslice1[j][0]
				c=latslice1[i][-1]
				d=latslice1[j][0]
				a1=longslice1[i].index[-1]
				b1=longslice1[j].index[0]
				c1=latslice1[i].index[-1]
				d1=latslice1[j].index[0]
			
				if (b1-a1).seconds != 0 and (d1-c1).seconds != 0:
					if (b-a) == 0 or (d-c) == 0:
						interlong.append(a)
						interlat.append(c)
					else:
						interlong.append(a+(((mybase[j]-a1).seconds/(b1-a1).seconds)*(b-a)))
						interlat.append(c+(((mybase[j]-c1).seconds/(d1-c1).seconds)*(d-c)))
				elif (b1-a1).seconds == 0 and (d1-c1).seconds != 0:
					interlong.append(a)
					interlat.append(c+(((mybase[j]-c1).seconds/(d1-c1).seconds)*(d-c)))
				elif (b1-a1).seconds != 0 and (d1-c1).seconds == 0:
					interlong.append(a+(((mybase[j]-a1).seconds/(b1-a1).seconds)*(b-a)))
					interlat.append(c)
				elif (b1-a1).seconds == 0 or (d1-c1).seconds == 0:
					interlong.append(a)
					interlat.append(c)

		#add the first element
		interlong1=[longslice1[0][0]]+interlong
		interlat1=[latslice1[0][0]]+interlat
		
		interdist=[]
		interspeed=[]
		#calculate distance and speed
		for i in range(0, int(arrinterval*z), int(arrinterval)):
			d1=interlong1[i: int(i+arrinterval)]
			d2=interlat1[i: int(i+arrinterval)]
			
			Rad = 6371.0
			for p, q, r,s in zip(d1, d2, d1[1:], d2[1:]):
				lat1 = radians(q)
				lon1 = radians(p)
				lat2 = radians(s)
				lon2 = radians(r)
				dlon = lon2 - lon1
				dlat = lat2 - lat1
				a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
				c = 2 * atan2(sqrt(a), sqrt(1-a))
				interdist.append(Rad * c) #distance
				interspeed.append((Rad * c)/30) #speed
		
		interdist1=num.array(interdist)
		interdist2=interdist1.reshape((interdist1.shape[0]/(arrinterval-1)),arrinterval-1)
		interspeed1=num.array(interspeed)
		interspeed2=interspeed1.reshape((interspeed1.shape[0]/(arrinterval-1)),arrinterval-1)

		#distance
		driver = gdal.GetDriverByName('GTiff')
		dst_file = '*****.tif'
		#dst_file = '*****.tif'
		dataset = driver.Create(dst_file, int(arrinterval)-1, z, 1, gdal.GDT_Float32, ) 
		#create new rasters as the standard raster
		#dataset.SetGeoTransform(stand_geotrans) #match the standard raster
		#dataset.SetProjection(stand_proj)
		dataset.GetRasterBand(1).WriteArray(interdist2)
		dataset.FlushCache()

		#speed
		driver = gdal.GetDriverByName('GTiff')
		dst_file = '*****.tif'
		#dst_file = '*****.tif'
		dataset = driver.Create(dst_file, int(arrinterval)-1, z, 1, gdal.GDT_Float32, ) 
		#create new rasters as the standard raster
		#dataset.SetGeoTransform(stand_geotrans) #match the standard raster
		#dataset.SetProjection(stand_proj)
		dataset.GetRasterBand(1).WriteArray(interspeed2)
		dataset.FlushCache()
		"""

	#convert abnormal records into shapefile
			
		#t=num.array(listlong[1:]).reshape((len(listlong[1:])/arrinterval),arrinterval)
		#u=num.array(listlat[1:]).reshape((len(listlat[1:])/arrinterval),arrinterval)
		
		for w in range(len(datediff)):
			#w=29 #select which date and get the slice
			startlong=mylongs.index.searchsorted(date_start[w]) #get the indiex of the start date
			endlong=mylongs.index.searchsorted(date_end[w]) #get the indiex of the end date
			speclong=mylongs.ix[startlong:endlong] #slice the date range
			speclong1=speclong.unstack()['Along']

			startlat=mylats.index.searchsorted(date_start[w]) #get the indiex of the start date
			endlat=mylats.index.searchsorted(date_end[w]) #get the indiex of the end date
			speclat=mylats.ix[startlat:endlat] #slice the date range
			speclat1=speclat.unstack()['Alat']
	
			exdate=speclong.unstack()['mydate']

			mylinestring=[]

			for m,n,p,q in zip(speclong1, speclat1, speclong1[1:], speclat1[1:]):
				mylinestring.append(LineString([(m,n),(p,q)]))
			
			#whole tracks
			
			
			#individual tracks
			"""
			schema = {'geometry': 'LineString','properties': {'track'+str(w): 'str'}}
			with fiona.open('*****.shp', 'w', 'ESRI Shapefile', schema) as layer:
				for line, ex in zip(mylinestring, exdate):
					elem = {}
					elem['geometry'] = mapping(line) 
					elem['properties'] = {'track'+str(w): str(ex)}
					layer.write(elem)
			"""
		myxfile = glob.glob('*****.shp')
		"""
		for k,w in zip(myxfile, range(len(datediff))):
			driver_v = ogr.GetDriverByName('ESRI Shapefile')
			inDataset = driver_v.Open(k, 0)
			inLayer = inDataset.GetLayer()
			inspatialReference = osgeo.osr.SpatialReference()
			inspatialReference.SetWellKnownGeogCS('WGS84') #.ImportFromEPSG(4326)
			coordTrans = osr.CoordinateTransformation(inspatialReference, outSpatialRef)

			a='*****.shp'
			outdataset = driver_v.CreateDataSource(a)
			
			outlayer = outdataset.CreateLayer('tra'+str(w), outSpatialRef, geom_type=ogr.wkbLineString) 
			inLayerDefn = inLayer.GetLayerDefn()
			for i in range(0, inLayerDefn.GetFieldCount()):
    				fieldDefn = inLayerDefn.GetFieldDefn(i)
    				outlayer.CreateField(fieldDefn)
			outLayerDefn = outlayer.GetLayerDefn()

			infeature = inLayer.GetNextFeature()
			#while infeature:
			#for i in range(0, outLayerDefn.GetFieldCount()):
			for feature in inLayer:
				geometry = feature.GetGeometryRef()
				geometry.Transform(coordTrans)
				outFeature = ogr.Feature(outLayerDefn)
				outFeature.SetGeometry(geometry)
				p=inLayerDefn.GetFieldDefn(0).GetNameRef()
				outFeature.SetField(outLayerDefn.GetFieldDefn(0).GetNameRef(), feature.GetField(p))
				outlayer.CreateFeature(outFeature)

			outFeature.Destroy()
			infeature.Destroy()
			#infeature = inLayer.GetNextFeature()
			inDataset.Destroy()
			outdataset.Destroy()
		"""

#track differences by text analysis (daily comparison)
		
	#loops for daily comparisons using text analysis
		tanalyze=[]
		#mytext=[]
		for i in range(0, int(arrinterval*z), int(arrinterval)): #calculate 100 days
		#for j in range (int(arrinterval)):
			s1 = mygeo7[i: int(i+arrinterval)]
			s2 = mygeo7[int(i+arrinterval): int(i+arrinterval+arrinterval)]
				
			for a,b in zip(s1,s2):
				tanalyze.append(edit_distance(a,b))
		
		#mytext.append(sum(tanalyze)) #sum the whole day differences		
		#tanalyze=[] #must clean the list for the next comparisons
	
		#convert to numpy array
		textana=num.array(tanalyze)
		#textana=num.array(tanalyze)
		textana1=textana.reshape((textana.shape[0]/arrinterval),arrinterval)
	
	#calculate by spherical distance
		Rad = 6371.0
		mydistance=[]
		for i in range(0, int(arrinterval*z), int(arrinterval)):
			splong1=longlo[i: int(i+arrinterval)]
			splat1=latla[i: int(i+arrinterval)]
			splong2=longlo[int(i+arrinterval): int(i+arrinterval+arrinterval)]
			splat2=latla[int(i+arrinterval): int(i+arrinterval+arrinterval)]
			for p, q, r,s in zip(splong1, splat1, splong2, splat2):
				lat1 = radians(q)
				lon1 = radians(p)
				lat2 = radians(s)
				lon2 = radians(r)
				dlon = lon2 - lon1
				dlat = lat2 - lat1
				a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
				c = 2 * atan2(sqrt(a), sqrt(1-a))
				mydistance.append(Rad * c)
		
		spana=num.array(mydistance)
		spana1=spana.reshape((spana.shape[0]/arrinterval),arrinterval)
	
		"""
	#loops for weekly comparison
		#get the day of the week
		dayweek=mybase.weekday+1 
		#Return the day of the week as an integer, where Monday is 0 and Sunday is 6, so add 1
		
		weekanaly=[]
		for i in range(0, int(arrinterval*z), int(arrinterval)): #calculate 100 days
		#for j in range (int(arrinterval)):
			s1 = mygeo7[i: int(i+arrinterval)]
			s2 = mygeo7[int(i+arrinterval*7): int(i+arrinterval*7+arrinterval)]
			
			for a, b in zip(s1,s2):
				weekanaly.append(edit_distance(a,b))

		#textana=num.append(num.array(weekanaly), num.zeros((1,)))
		textana=num.array(weekanaly)
		textana1=textana.reshape((textana.shape[0]/arrinterval),arrinterval)
		"""
		
	#Calculate the alignment vector and corresponding distance
		#different benchmark
		
		"""
		montext=num.mean(textana1[0:30], axis=0) #the first month average
		monsp=num.mean(spana1[0:30], axis=0)
		tottext=num.mean(textana1, axis=0) #total average
		totsp=num.mean(spana1, axis=0)
		
		#remove outliers and average the rest of values
		retextana1=textana1*1
		masktext=textana1*1
		textmean=num.mean(retextana1, axis=0)
		textstd=num.std(retextana1, axis=0)
		for i in range(retextana1.shape[1]): #how many columns
			m=[row[i] for row in retextana1] #get each row by the loop
			for j in range(retextana1.shape[0]): #how many rows
				if m[j]>(textmean[i]+textstd[i]) or m[j] < (textmean[i]-textstd[i]): #outlier =0
					retextana1[j][i]=0
				else:
					masktext[j][i]=0
		
		nouttext=num.ma.array(retextana1, mask=masktext).mean(axis=0)
		outtext=num.array(nouttext)
		
		respana1=spana1*1
		masksp=spana1*1
		spmean=num.mean(respana1, axis=0)
		spstd=num.std(respana1, axis=0)
		for i in range(respana1.shape[1]): #how many columns
			m=[row[i] for row in respana1] #get each row by the loop
			for j in range(respana1.shape[0]): #how many rows
				if m[j]>(spmean[i]+spstd[i]) or m[j] < (spmean[i]-spstd[i]): #outlier =0
					respana1[j][i]=0
				else:
					masksp[j][i]=0

		noutsp=num.ma.array(respana1, mask=masksp).mean(axis=0)
		outsp=num.array(noutsp)
		
		"""
		"""
		DTWtext = []
		for g in range(textana1.shape[0]):
			#textana1[i]
			alignment = R.dtw(textana1[g], textana1[0], keep=True)
			dist = alignment.rx('distance')[0][0]
			nordist = alignment.rx('normalizedDistance')[0][0] 
			DTWtext.append(nordist)
		
		DTWsp = []
		for h in range(spana1.shape[0]):
			#textana1[i]
			alignment = R.dtw(spana1[h], spana1[0], keep=True)
			dist = alignment.rx('distance')[0][0]
			nordist = alignment.rx('normalizedDistance')[0][0] 
			DTWsp.append(nordist)
		
		"""
#graphs export
	#convert numpy array to tif and export
		"""
		#text analysis
		driver_ra = gdal.GetDriverByName('GTiff')
		dst_file = '*****.tif'
		#dst_file = '*****.tif'
		dataset = driver_ra.Create(dst_file, int(arrinterval), z, 1, gdal.GDT_Float32, ) 
		#create new rasters as the standard raster
		#dataset.SetGeoTransform(stand_geotrans) #match the standard raster
		#dataset.SetProjection(stand_proj)
		dataset.GetRasterBand(1).WriteArray(textana1)
		dataset.FlushCache()
		
		#spherical distance
		driver_ra = gdal.GetDriverByName('GTiff')
		dst_file = '*****.tif'
		#dst_file = '*****.tif'
		dataset = driver_ra.Create(dst_file, int(arrinterval), z, 1, gdal.GDT_Float32, ) 
		#create new rasters as the standard raster
		#dataset.SetGeoTransform(stand_geotrans) #match the standard raster
		#dataset.SetProjection(stand_proj)
		dataset.GetRasterBand(1).WriteArray(spana1)
		dataset.FlushCache()
		"""
			
	#DTW comparisons as a csv file
		#numpy.savetxt('*****.csv', DTWtext, delimiter=",")
		#numpy.savetxt('*****.csv', DTWsp, delimiter=",")
		#numpy.savetxt('*****.csv', myDTW, delimiter=",")
	
		
except psycopg2.DatabaseError, e:
#In case of an error, the function roll back any possible changes to our database table.
	if con: 
		con.rollback()
	
	print 'Error %s' % e    
	sys.exit(1)
    
    
finally: #release the resources
    
	if con:
        	con.close()
