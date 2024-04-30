# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:19:54 2024

@author: Kailah Gordon
"""
#Case Study for Lufthansa Flight 469 which encountered SVR Turbulence causing multiple injuries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.feature as cf
import cartopy.crs as ccrs
import pygrib
import metpy
from datetime import datetime

#data downloaded from amazon web services
data1 = pygrib.open ('hrrr.t00z.wrfprsf00 (1).grib2')
#data1.select() to list the data

#Map 1 - 500mb Geopotential Heights and Winds------------------------------------
#Select data only related to 500mb geopotential height and 500mb winds
data1.select(name = 'Geopotential height')
data1.select(name = 'U component of wind')
data1.select(name = 'V component of wind')
#Use selected data to pull the values for height and wind
hgt500 = data1[253]; hgt = hgt500['values']
uwind = data1[259]; uw = uwind['values']
vwind = data1[260]; vw = vwind['values']

#calculate for the magnitude of the wind
mag = np.sqrt((uw**2)+(vw**2))
#set lat lons to one of the datasets
lats, lons = hgt500.latlons()
#plot the map focused on CONUS 
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-97.,central_latitude=(30.0+60.0)/2)
ax=plt.axes(projection=proj)
#create the base map and set the extents for CONUS, add in base features
ax.set_extent([-119.90,-73.50,23.08,50.00])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightsteelblue')
ax.add_feature(cf.COASTLINE,edgecolor='dimgray')
ax.add_feature(cf.STATES,edgecolor='dimgray')
ax.add_feature(cf.BORDERS,edgecolor='dimgray',linestyle='-')
ax.add_feature(cf.LAKES,color='lightsteelblue', alpha=0.5)
#set grid lines to TRUE and set bounds for the color bar
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.xlabel_style = {'rotation':45}
#set the bounds to only include the necessary values for wind speed
bounds = [30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120]

gl.top_labels = False
gl.left_labels = False
#plot the wind speed in knots 
map=plt.contourf(lons,lats,mag*1.944,bounds, cmap=plt.cm.afmhot_r,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('knots')
#plot the height lines and wind barbs
h=plt.contour (lons, lats, hgt, np.arange(np.min(hgt), np.max(hgt),40), linestyles='-', linewidths=2, colors='black', transform=ccrs.PlateCarree())
plt.barbs(lons[::80,::80],lats[::80,::80],uw[::80,::80],vw[::80,::80],transform=ccrs.PlateCarree())
#plot a point where the event occurred 
plt.plot(-89.45,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())
#create a title and save figure as a png
plt.title ('March 02 00Z HRRR Forecast Hour 00Z \n500 mb Heights (m) / 500 mb Winds (knots)')
plt.savefig('3.02.HeightWindCONUS.png')
plt.close()

#Map 2 - 500 mb Geopotential Heights and Winds (regional/case specific)-----------
#set lat lons to one of the datasets
lats, lons = hgt500.latlons()
#plot the map focused on region of case study 
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-89.40,central_latitude=35.06)
ax=plt.axes(projection=proj)
#create a base map with the extent focused over the region in question
ax.set_extent([-92.40,-86.40,38.06,32.06])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightsteelblue')
ax.add_feature(cf.COASTLINE,edgecolor='dimgray')
ax.add_feature(cf.STATES,edgecolor='dimgray')
ax.add_feature(cf.BORDERS,edgecolor='dimgray',linestyle='-')
ax.add_feature(cf.LAKES,color='lightsteelblue', alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.xlabel_style = {'rotation':45}

#set the bounds to only include the necessary values for wind speed
bounds = [30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120]

gl.top_labels = False
gl.left_labels = False

#plot the wind speed in knots
map=plt.contourf(lons,lats,mag*1.944,bounds, cmap=plt.cm.afmhot_r,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('knots')
#plot the height lines and point of interest 
h=plt.contour (lons, lats, hgt, np.arange(np.min(hgt), np.max(hgt),40), linestyles='-', linewidths=2, colors='black', transform=ccrs.PlateCarree())

plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())
#create a title and save the figure
plt.title ('March 02 00Z HRRR Forecast Hour 00Z \n500 mb Heights (m) / 500 mb Winds (knots)')
plt.savefig('3.02.HeightWindRegion.png')
plt.close()

#Map 3 - 500mb Temperature------------------------------------------------------
#select data for temperature
data1.select(name = 'Temperature')

#Use selected data to pull the values
tmp500 = data1[254]; tmp = tmp500['values']

#Convert temperature from Kelvin to Celsius
tmpC = tmp-273.15
#set lat lons to one of the datasets
lats, lons = tmp500.latlons()
#plot the map focused on CONUS 
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-97.,central_latitude=(30.0+60.0)/2)
ax=plt.axes(projection=proj)
#create base map for CONUS
ax.set_extent([-119.90,-73.50,23.08,50.00])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightsteelblue')
ax.add_feature(cf.COASTLINE,edgecolor='black')
ax.add_feature(cf.STATES,edgecolor='black')
ax.add_feature(cf.BORDERS,edgecolor='black',linestyle='-')
ax.add_feature(cf.LAKES,color='lightsteelblue', alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.xlabel_style = {'rotation':45}
#Set the bounds for temperature 
bounds = [-40,-37.5,-35,-32.5,-30,-27.5,-25,-22.5,-20,-17.5,-15,-12.5,-10,-7.5,-5,-2.5,0]

gl.top_labels = False
gl.left_labels = False
#plot the contours for temperature and the point of interest
map=plt.contourf(lons,lats,tmpC,bounds, cmap=plt.cm.turbo,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('Celsius')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())
#create a title and save the figure
plt.title ('March 02 00Z HRRR Forecast Hour 00Z \n500 mb Temperature (C)')
plt.savefig('3.02.TemperatureCONUS.png')
plt.close()

#Map 4 - 500 mb Temperature Regional--------------------------------------------
#follow the steps from Map 3 bu for the regional study
#set lat lons to one of the datasets
lats, lons = tmp500.latlons()
#plot the map focused on region of case study 
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-89.40,central_latitude=35.06)
ax=plt.axes(projection=proj)

ax.set_extent([-92.40,-86.40,38.06,32.06])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightsteelblue')
ax.add_feature(cf.COASTLINE,edgecolor='black')
ax.add_feature(cf.STATES,edgecolor='black')
ax.add_feature(cf.BORDERS,edgecolor='black',linestyle='-')
ax.add_feature(cf.LAKES,color='lightsteelblue', alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.xlabel_style = {'rotation':45}

bounds = [-40,-37.5,-35,-32.5,-30,-27.5,-25,-22.5,-20,-17.5,-15,-12.5,-10,-7.5,-5,-2.5,0]

gl.top_labels = False
gl.left_labels = False

map=plt.contourf(lons,lats,tmpC,bounds, cmap=plt.cm.turbo,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('Celsius')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())

plt.title ('March 02 00Z HRRR Forecast Hour 00Z \n500 mb Temperature (C)')
plt.savefig('3.02.TemperatureRegion.png')
plt.close()

#Map 5- TKE------------------------------------------------------------------------
#select data for the 3 wind components to use to calculate TKE
data1.select(name = 'U component of wind')
data1.select(name = 'V component of wind')
data1.select(name = 'Vertical velocity') # w component of the wind
#set the values for the selected data
uwind = data1[259]; uw = uwind['values']
vwind = data1[260]; vw = vwind['values']
wwind = data1[258]; ww = wwind['values']
#set the lat,lon for wind components
lats, lons = uwind.latlons()

#calculate TKE using metpy- produces a 1D array- need to turn it to a 2d array for plot
#only working with one hour of data so the TKE is averaged over this one hour, not instantaneous
from metpy.calc import tke
from metpy.units import units
TKE = metpy.calc.tke(uw, vw, ww, perturbation=False, axis=-1)
#reshape into a 2d array using the latitude and longitude 
TKE2d = np.broadcast_to(TKE[:, np.newaxis], lats.shape)

#plot the map focused on region of case study 
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-89.40,central_latitude=35.06)
ax=plt.axes(projection=proj)
#set the vounds for the regional study and create the base map
ax.set_extent([-92.40,-86.40,38.06,32.06])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightsteelblue')
ax.add_feature(cf.COASTLINE,edgecolor='black')
ax.add_feature(cf.STATES,edgecolor='black')
ax.add_feature(cf.BORDERS,edgecolor='black',linestyle='-')
ax.add_feature(cf.LAKES,color='lightsteelblue', alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.xlabel_style = {'rotation':45}

gl.top_labels = False
gl.left_labels = False
#plot TKE and the point location
map=plt.contourf(lons,lats,TKE2d,np.arange(np.min(TKE),np.max(TKE),.10), cmap=plt.cm.gist_ncar,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('TKE (m2/s2)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())

plt.title ('March 02 00Z HRRR Forecast Hour 00Z \nTurbulent Kinetic Energy (m2/s2)')
plt.savefig('3.02.TKERegion.png')
plt.close()

#Map 6 - Skew T------------------------------------------------------------
from datetime import datetime
from metpy.plots import SkewT, Hodograph
from metpy.units import pandas_dataframe_to_unit_arrays, units
from scipy.interpolate import interp1d
from siphon.simplewebservice.wyoming import WyomingUpperAir
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import sys
# Get the data we want
df = WyomingUpperAir.request_data(datetime(2023, 3, 2, 0), 'LZK')
sounding = pandas_dataframe_to_unit_arrays(df)
# Calculate thermodynamics
lcl_pressure, lcl_temperature = mpcalc.lcl(sounding['pressure'][0],
                                           sounding['temperature'][0],
                                           sounding['dewpoint'][0])

lfc_pressure, lfc_temperature = mpcalc.lfc(sounding['pressure'],
                                           sounding['temperature'],
                                           sounding['dewpoint'])

el_pressure, el_temperature = mpcalc.el(sounding['pressure'],
                                        sounding['temperature'],
                                        sounding['dewpoint'])

parcel_profile = mpcalc.parcel_profile(sounding['pressure'],
                                       sounding['temperature'][0],
                                       sounding['dewpoint'][0])
# Some new imports
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from metpy.plots import add_metpy_logo
# Make the plot
# Create a new figure. The dimensions here give a good aspect ratio
fig = plt.figure(figsize=(9, 9))

# Grid for plots
gs = gridspec.GridSpec(3, 3)
skew = SkewT(fig, rotation=45, subplot=gs[:, :2])
metpy.plots.add_metpy_logo(fig, 450,50,size='small')
# Plot the sounding using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(sounding['pressure'], sounding['temperature'], 'tab:red')
skew.plot(sounding['pressure'], sounding['dewpoint'], 'tab:green')
skew.plot(sounding['pressure'], parcel_profile, 'k')

# Mask barbs to be below 100 hPa only
mask = sounding['pressure'] >= 100 * units.hPa
skew.plot_barbs(sounding['pressure'][mask], sounding['u_wind'][mask], sounding['v_wind'][mask])
skew.ax.set_ylim(1000, 100)

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

# Good bounds for aspect ratio
skew.ax.set_xlim(-30, 40)

if lcl_pressure:
    skew.ax.plot(lcl_temperature, lcl_pressure, marker="_", color='tab:orange', markersize=30, markeredgewidth=3)
    
if lfc_pressure:
    skew.ax.plot(lfc_temperature, lfc_pressure, marker="_", color='tab:brown', markersize=30, markeredgewidth=3)
    
if el_pressure:
    skew.ax.plot(el_temperature, el_pressure, marker="_", color='tab:blue', markersize=30, markeredgewidth=3)
plt.title('LZK 3/2/2023 00Z Sounding', loc='left', fontsize = 20)
# Create a hodograph
agl = sounding['height'] - sounding['height'][0]
mask = agl <= 10 * units.km
intervals = np.array([0, 1, 3, 5, 8]) * units.km
colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:olive']
ax = fig.add_subplot(gs[0, -1])
h = Hodograph(ax, component_range=60.)
h.add_grid(increment=20)
h.plot_colormapped(sounding['u_wind'][mask], sounding['v_wind'][mask], agl[mask], intervals=intervals, colors=colors)
plt.savefig('3.2.23.SkewT.png')
plt.close()

# Map 7 - Gradient Richardson Number----------------------------------------------------
#values less than 0.25 indicate turbulence
data1.select(name = 'U component of wind')
data1.select(name = 'V component of wind')
data1.select(name = 'Potential temperature')
data1.select(name = 'Geopotential height')

#set the values for the selected data
uwind = data1[259]; uw = uwind['values']
vwind = data1[260]; vw = vwind['values']
ptmp = data1[617]; pt = wwind['values']
hgt500 = data1[253]; hgt = hgt500['values']

lats, lons = hgt500.latlons()

#calculate the richardson number
rn = metpy.calc.gradient_richardson_number(hgt * units('m'), pt * units('K'), uw * units('m/s'), vw * units('m/s'), vertical_dim=0)
invalid_indices = np.isnan(rn)
rn_clean = np.ma.masked_invalid(rn)

southeast = (lons > -92.0) & (lons < -86.0)
filtered_lons = lons[southeast]
filtered_rn_clean = rn_clean[southeast]

# Filter Richardson number values within the range -100 to 100
rn_range_indices = (filtered_rn_clean >= -20) & (filtered_rn_clean <= 10)
filtered_lons_range = filtered_lons[rn_range_indices]
filtered_rn_clean_range = filtered_rn_clean[rn_range_indices]

x = filtered_lons_range
y = filtered_rn_clean_range

#plot the richardsn number as a line plot
fig = plt.figure(figsize = (8,6))
plt.plot(x,y)
plt.grid(True)
plt.xlabel('Longitude')
plt.ylabel('Richardson Number')
plt.title('March 02 00Z HRRR Forecast Hour 00Z \n500 mb Richardson Number by Regional Longitude')

plt.savefig('3.02.RichardsonNumberGraph.png')
plt.close()

#Map 9 - 850-500mb Vertical Shear -------------------------------------------------------------------------------
#Select the data for the 500 mb winds and the 850mb 
data1.select(name = 'U component of wind')
data1.select(name = 'V component of wind')
#Use selected data to pull the values
uwind500 = data1[259]; uw500 = uwind500['values']
vwind500 = data1[260]; vw500 = vwind500['values']
uwind850 = data1[455]; uw850 = uwind850['values']
vwind850 = data1[456]; vw850 = vwind850['values']

#calculate for the shear of the wind
shear = np.sqrt((uw500-uw850)**2 +(vw500-vw850)**2)
#set the latitude/longitude
lats, lons = uwind500.latlons()
#set the figure size
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-97.,central_latitude=(30.0+60.0)/2)
ax=plt.axes(projection=proj)
#create the base map and set the extents for CONUS, add in base features
ax.set_extent([-119.90,-73.50,23.08,50.00])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightsteelblue')
ax.add_feature(cf.COASTLINE,edgecolor='dimgray')
ax.add_feature(cf.STATES,edgecolor='dimgray')
ax.add_feature(cf.BORDERS,edgecolor='dimgray',linestyle='-')
ax.add_feature(cf.LAKES,color='lightsteelblue', alpha=0.5)
#set grid lines to TRUE and set bounds for the color bar
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.xlabel_style = {'rotation':45}
#Set the bounds for wind shear
bounds = [0,5,10,15,20,25,30,35,40,45,50]

gl.top_labels = False
gl.left_labels = False
#plot the wind shear and specific point
map=plt.contourf(lons,lats,shear,bounds, cmap=plt.cm.jet,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('Wind Shear m/s')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())

plt.title ('March 02 00Z HRRR Forecast Hour 00Z \n850 mb to 500 mb Vertical Wind Shear')
plt.savefig('3.02.00z.WindShear.png')
plt.close()

#Wind shear regional---------------------------------------------------------------------------
#calculate for the shear of the wind
#Folllow steps from previous map
shear = np.sqrt((uw500-uw850)**2 +(vw500-vw850)**2)
#set the latitude/longitude
lats, lons = uwind500.latlons()
#set the figure size
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-89.40,central_latitude=35.06)
ax=plt.axes(projection=proj)
#create the base map and set the extents for CONUS, add in base features
ax.set_extent([-92.40,-86.40,38.06,32.06])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightsteelblue')
ax.add_feature(cf.COASTLINE,edgecolor='dimgray')
ax.add_feature(cf.STATES,edgecolor='dimgray')
ax.add_feature(cf.BORDERS,edgecolor='dimgray',linestyle='-')
ax.add_feature(cf.LAKES,color='lightsteelblue', alpha=0.5)
#set grid lines to TRUE and set bounds for the color bar
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.xlabel_style = {'rotation':45}

bounds = [0,5,10,15,20,25,30,35,40,45,50]

gl.top_labels = False
gl.left_labels = False

map=plt.contourf(lons,lats,shear,bounds, cmap=plt.cm.jet,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('Wind Shear m/s')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())

plt.title ('March 02 00Z HRRR Forecast Hour 00Z \n850 mb to 500 mb Vertical Wind Shear Regional')
plt.savefig('3.02.00z.WindShearRegional.png')
plt.close()

#read in second data file
data2 = pygrib.open ('hrrr.t23z.wrfprsf00 (1).grib2')
#select data
data2.select(name = 'U component of wind')
data2.select(name = 'V component of wind')
#Use selected data to pull the values
uwind500 = data2[259]; uw500 = uwind500['values']
vwind500 = data2[260]; vw500 = vwind500['values']
uwind850 = data2[455]; uw850 = uwind850['values']
vwind850 = data2[456]; vw850 = vwind850['values']

#calculate for the shear of the wind
shear = np.sqrt((uw500-uw850)**2 +(vw500-vw850)**2)
#set the latitude/longitude
lats, lons = uwind500.latlons()
#set the figure size
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-97.,central_latitude=(30.0+60.0)/2)
ax=plt.axes(projection=proj)
#create the base map and set the extents for CONUS, add in base features
ax.set_extent([-119.90,-73.50,23.08,50.00])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightsteelblue')
ax.add_feature(cf.COASTLINE,edgecolor='dimgray')
ax.add_feature(cf.STATES,edgecolor='dimgray')
ax.add_feature(cf.BORDERS,edgecolor='dimgray',linestyle='-')
ax.add_feature(cf.LAKES,color='lightsteelblue', alpha=0.5)
#set grid lines to TRUE and set bounds for the color bar
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.xlabel_style = {'rotation':45}

bounds = [0,5,10,15,20,25,30,35,40,45,50]

gl.top_labels = False
gl.left_labels = False

map=plt.contourf(lons,lats,shear,bounds, cmap=plt.cm.jet,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('Wind Shear m/s')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())

plt.title ('March 01 23Z HRRR Forecast Hour 00Z \n850 mb to 500 mb Vertical Wind Shear')
plt.savefig('3.01.23z.WindShear.png')
plt.close()

shear = np.sqrt((uw500-uw850)**2 +(vw500-vw850)**2)
#set the latitude/longitude
lats, lons = uwind500.latlons()
#set the figure size
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-89.40,central_latitude=35.06)
ax=plt.axes(projection=proj)
#create the base map and set the extents for CONUS, add in base features
ax.set_extent([-92.40,-86.40,38.06,32.06])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightsteelblue')
ax.add_feature(cf.COASTLINE,edgecolor='dimgray')
ax.add_feature(cf.STATES,edgecolor='dimgray')
ax.add_feature(cf.BORDERS,edgecolor='dimgray',linestyle='-')
ax.add_feature(cf.LAKES,color='lightsteelblue', alpha=0.5)
#set grid lines to TRUE and set bounds for the color bar
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.xlabel_style = {'rotation':45}

bounds = [0,5,10,15,20,25,30,35,40,45,50]

gl.top_labels = False
gl.left_labels = False

map=plt.contourf(lons,lats,shear,bounds, cmap=plt.cm.jet,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('Wind Shear m/s')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())

plt.title ('March 01 23Z HRRR Forecast Hour 00Z \n850 mb to 500 mb Vertical Wind Shear Regional')
plt.savefig('3.01.23z.WindShearRegional.png')
plt.close()

#read in data file three
data3 = pygrib.open ('hrrr.t22z.wrfprsf00.grib2')

data3.select(name = 'U component of wind')
data3.select(name = 'V component of wind')
#Use selected data to pull the values
uwind500 = data3[259]; uw500 = uwind500['values']
vwind500 = data3[260]; vw500 = vwind500['values']
uwind850 = data3[455]; uw850 = uwind850['values']
vwind850 = data3[456]; vw850 = vwind850['values']

#calculate for the shear of the wind
shear = np.sqrt((uw500-uw850)**2 +(vw500-vw850)**2)
#set the latitude/longitude
lats, lons = uwind500.latlons()
#set the figure size
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-97.,central_latitude=(30.0+60.0)/2)
ax=plt.axes(projection=proj)
#create the base map and set the extents for CONUS, add in base features
ax.set_extent([-119.90,-73.50,23.08,50.00])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightsteelblue')
ax.add_feature(cf.COASTLINE,edgecolor='dimgray')
ax.add_feature(cf.STATES,edgecolor='dimgray')
ax.add_feature(cf.BORDERS,edgecolor='dimgray',linestyle='-')
ax.add_feature(cf.LAKES,color='lightsteelblue', alpha=0.5)
#set grid lines to TRUE and set bounds for the color bar
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.xlabel_style = {'rotation':45}

bounds = [0,5,10,15,20,25,30,35,40,45,50]

gl.top_labels = False
gl.left_labels = False

map=plt.contourf(lons,lats,shear,bounds, cmap=plt.cm.jet,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('Wind Shear m/s')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())

plt.title ('March 01 22Z HRRR Forecast Hour 00Z \n8500 mb to 500 mb Vertical Wind Shear')
plt.savefig('3.01.22z.WindShear.png')
plt.close()

shear = np.sqrt((uw500-uw850)**2 +(vw500-vw850)**2)
#set the latitude/longitude
lats, lons = uwind500.latlons()
#set the figure size
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-89.40,central_latitude=35.06)
ax=plt.axes(projection=proj)
#create the base map and set the extents for CONUS, add in base features
ax.set_extent([-92.40,-86.40,38.06,32.06])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightsteelblue')
ax.add_feature(cf.COASTLINE,edgecolor='dimgray')
ax.add_feature(cf.STATES,edgecolor='dimgray')
ax.add_feature(cf.BORDERS,edgecolor='dimgray',linestyle='-')
ax.add_feature(cf.LAKES,color='lightsteelblue', alpha=0.5)
#set grid lines to TRUE and set bounds for the color bar
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.xlabel_style = {'rotation':45}

bounds = [0,5,10,15,20,25,30,35,40,45,50]

gl.top_labels = False
gl.left_labels = False

map=plt.contourf(lons,lats,shear,bounds, cmap=plt.cm.jet,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('Wind Shear m/s')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())

plt.title ('March 01 22Z HRRR Forecast Hour 00Z \n850 mb to 500 mb Vertical Wind Shear Regional')
plt.savefig('3.01.22z.WindShearRegional.png')
plt.close()

#read in data file 4
data4 = pygrib.open ('hrrr.t21z.wrfprsf00.grib2')

data4.select(name = 'U component of wind')
data4.select(name = 'V component of wind')
#Use selected data to pull the values
uwind500 = data4[259]; uw500 = uwind500['values']
vwind500 = data4[260]; vw500 = vwind500['values']
uwind850 = data4[455]; uw850 = uwind850['values']
vwind850 = data4[456]; vw850 = vwind850['values']

#calculate for the shear of the wind
shear = np.sqrt((uw500-uw850)**2 +(vw500-vw850)**2)
#set the latitude/longitude
lats, lons = uwind500.latlons()
#set the figure size
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-97.,central_latitude=(30.0+60.0)/2)
ax=plt.axes(projection=proj)
#create the base map and set the extents for CONUS, add in base features
ax.set_extent([-119.90,-73.50,23.08,50.00])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightsteelblue')
ax.add_feature(cf.COASTLINE,edgecolor='dimgray')
ax.add_feature(cf.STATES,edgecolor='dimgray')
ax.add_feature(cf.BORDERS,edgecolor='dimgray',linestyle='-')
ax.add_feature(cf.LAKES,color='lightsteelblue', alpha=0.5)
#set grid lines to TRUE and set bounds for the color bar
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.xlabel_style = {'rotation':45}

bounds = [0,5,10,15,20,25,30,35,40,45,50]

gl.top_labels = False
gl.left_labels = False

map=plt.contourf(lons,lats,shear,bounds, cmap=plt.cm.jet,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('Wind Shear m/s')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())

plt.title ('March 01 21Z HRRR Forecast Hour 00Z \n850 mb to 500 mb Vertical Wind Shear')
plt.savefig('3.01.21z.WindShear.png')
plt.close()

shear = np.sqrt((uw500-uw850)**2 +(vw500-vw850)**2)
#set the latitude/longitude
lats, lons = uwind500.latlons()
#set the figure size
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-89.40,central_latitude=35.06)
ax=plt.axes(projection=proj)
#create the base map and set the extents for CONUS, add in base features
ax.set_extent([-92.40,-86.40,38.06,32.06])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightsteelblue')
ax.add_feature(cf.COASTLINE,edgecolor='dimgray')
ax.add_feature(cf.STATES,edgecolor='dimgray')
ax.add_feature(cf.BORDERS,edgecolor='dimgray',linestyle='-')
ax.add_feature(cf.LAKES,color='lightsteelblue', alpha=0.5)
#set grid lines to TRUE and set bounds for the color bar
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.xlabel_style = {'rotation':45}

bounds = [0,5,10,15,20,25,30,35,40,45,50]

gl.top_labels = False
gl.left_labels = False

map=plt.contourf(lons,lats,shear,bounds, cmap=plt.cm.jet,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('Wind Shear m/s')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())

plt.title ('March 01 21Z HRRR Forecast Hour 00Z \n850 mb to 500 mb Vertical Wind Shear Regional')
plt.savefig('3.01.21z.WindShearRegional.png')
plt.close()

data5 = pygrib.open ('hrrr.t20z.wrfprsf00.grib2')

data5.select(name = 'U component of wind')
data5.select(name = 'V component of wind')
#Use selected data to pull the values
uwind500 = data5[259]; uw500 = uwind500['values']
vwind500 = data5[260]; vw500 = vwind500['values']
uwind850 = data5[455]; uw850 = uwind850['values']
vwind850 = data5[456]; vw850 = vwind850['values']

#calculate for the shear of the wind
shear = np.sqrt((uw500-uw850)**2 +(vw500-vw850)**2)
#set the latitude/longitude
lats, lons = uwind500.latlons()
#set the figure size
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-97.,central_latitude=(30.0+60.0)/2)
ax=plt.axes(projection=proj)
#create the base map and set the extents for CONUS, add in base features
ax.set_extent([-119.90,-73.50,23.08,50.00])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightsteelblue')
ax.add_feature(cf.COASTLINE,edgecolor='dimgray')
ax.add_feature(cf.STATES,edgecolor='dimgray')
ax.add_feature(cf.BORDERS,edgecolor='dimgray',linestyle='-')
ax.add_feature(cf.LAKES,color='lightsteelblue', alpha=0.5)
#set grid lines to TRUE and set bounds for the color bar
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.xlabel_style = {'rotation':45}

bounds = [0,5,10,15,20,25,30,35,40,45,50]

gl.top_labels = False
gl.left_labels = False

map=plt.contourf(lons,lats,shear,bounds, cmap=plt.cm.jet,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('Wind Shear m/s')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())

plt.title ('March 01 20Z HRRR Forecast Hour 00Z \n850 mb to 500 mb Vertical Wind Shear')
plt.savefig('3.01.20z.WindShear.png')
plt.close()
#Make regional map
shear = np.sqrt((uw500-uw850)**2 +(vw500-vw850)**2)
#set the latitude/longitude
lats, lons = uwind500.latlons()
#set the figure size
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-89.40,central_latitude=35.06)
ax=plt.axes(projection=proj)
#create the base map and set the extents for CONUS, add in base features
ax.set_extent([-92.40,-86.40,38.06,32.06])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN,color='lightsteelblue')
ax.add_feature(cf.COASTLINE,edgecolor='dimgray')
ax.add_feature(cf.STATES,edgecolor='dimgray')
ax.add_feature(cf.BORDERS,edgecolor='dimgray',linestyle='-')
ax.add_feature(cf.LAKES,color='lightsteelblue', alpha=0.5)
#set grid lines to TRUE and set bounds for the color bar
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.xlabel_style = {'rotation':45}

bounds = [0,5,10,15,20,25,30,35,40,45,50]

gl.top_labels = False
gl.left_labels = False

map=plt.contourf(lons,lats,shear,bounds, cmap=plt.cm.jet,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('Wind Shear m/s')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())

plt.title ('March 01 20Z HRRR Forecast Hour 00Z \n850 mb to 500 mb Vertical Wind Shear Regional')
plt.savefig('3.01.20z.WindShearRegional.png')
plt.close()

#Map 11 - RADAR- SHOWS RAIN BUT TURB WAS 37,000ft Beam Height only 2.2k feet at location---------------------------------------------------------------------------------------
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from metpy.calc import azimuth_range_to_lat_lon
from metpy.cbook import get_test_data
from metpy.io import Level2File
from metpy.plots import add_metpy_logo, add_timestamp, USCOUNTIES
from metpy.units import units

###########################################
# Open the file
file1 = 'KNQA20230301_230141_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar1.png') 
plt.close()

# Open the file
file1 = 'KNQA20230301_230715_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar2.png') 
plt.close()

# Open the file
file1 = 'KNQA20230301_231325_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar3.png') 
plt.close()

# Open the file
file1 = 'KNQA20230301_231934_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar4.png') 
plt.close()

# Open the file
file1 = 'KNQA20230301_232543_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar5.png') 
plt.close()

# Open the file
file1 = 'KNQA20230301_233152_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar6.png') 
plt.close()

# Open the file
file1 = 'KNQA20230301_233749_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar7.png') 
plt.close()

# Open the file
file1 = 'KNQA20230301_234358_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar8.png') 
plt.close()

# Open the file
file1 = 'KNQA20230301_234939_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar9.png') 
plt.close()

# Open the file
file1 = 'KNQA20230301_235513_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar10.png') 
plt.close()

# Open the file
file1 = 'KNQA20230302_000102_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar11.png') 
plt.close()

# Open the file
file1 = 'KNQA20230302_001244_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar12.png') 
plt.close()

# Open the file
file1 = 'KNQA20230302_001826_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar13.png') 
plt.close()

# Open the file
file1 = 'KNQA20230302_002400_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar14.png') 
plt.close()

# Open the file
file1 = 'KNQA20230302_002933_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar15.png') 
plt.close()

# Open the file
file1 = 'KNQA20230302_003507_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar16.png') 
plt.close()

# Open the file
file1 = 'KNQA20230302_004104_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar17.png') 
plt.close()

# Open the file
file1 = 'KNQA20230302_004703_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar18.png') 
plt.close()

# Open the file
file1 = 'KNQA20230302_005301_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar19.png') 
plt.close()

# Open the file
file1 = 'KNQA20230302_005911_V06'
rad1 = Level2File(file1)

print(rad1.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in rad1.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = rad1.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in rad1.sweeps[sweep]])

rho_hdr = rad1.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in rad1.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = rad1.sweeps[0][0][1].lon
cent_lat = rad1.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)
    bounds = [-20,-10,0,10,20,30,40,50,60,70]
    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    mesh = ax.pcolormesh(xlocs, ylocs, data, cmap='jet', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,y_inline=False,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation':45}
    gl.top_labels = False
    gl.left_labels = False
    add_timestamp(ax, rad1.dt, y=0.02, high_contrast=True)
    cbar = plt.colorbar(mesh, location='bottom')
    cbar.set_label ('Reflectivity (dbZ)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 20, transform = ccrs.PlateCarree())

plt.title('Radar Reflectivity \nMarch 1 23Z to March 2 1z')
plt.savefig('Radar20.png') 
plt.close()

#making a gif of the radar images
#Python script to convert PNG files to an ANIMATED GIF file.  Make sure that PNG
#files are of nearly equal size. Alternatively, you can specify size of GIF file
#[H x W] = Height x Width.  Apply argument -s to scale [rescale] the images.  To
#disable scaling, set n < 0 in case you are code is run from command line.
#-------------------------------------------------------------------------------
import sys, os
from PIL import Image
import glob

fGIF = "Radar.gif"
H = 1152
W = 1152
n = 1
# Create the frames
frames = []
images = glob.glob("Radar*.png")
#resize the images 
for i in images:
    newImg = Image.open(i)
    if (len(sys.argv) < 2 and n > 0):
        newImg = newImg.resize((W, H))
    frames.append(newImg)
 
# Save into a GIF file that loops forever: duration is in milli-second
frames[0].save(fGIF, format='GIF', append_images=frames[1:],
    save_all=True, duration=700, loop=0)

