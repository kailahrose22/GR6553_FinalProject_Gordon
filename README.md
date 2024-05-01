# Clear Air Turbulence Analysis Script

## Project Despription
For my final project, I wanted to analyze a case study of Severe Clear Air Turbulence (CAT) that occurred on March 1, 2023, using figures created through a Python Code. 
The script found on this page works to create many figures for analysis using High-Resolution Rapid Refresh (HRRR) model data. The data was downloaded from [HRRR Data](https://noaa-hrrr-bdp-pds.s3.amazonaws.com/index.html). 
The script also utilizes NEXRAD Level II Radar data which can be found at [NEXRAD Data](https://s3.amazonaws.com/noaa-nexrad-level2/index.html).
The figures created for analysis in this project visualize many meteorological variables including 500mb Geopotential Height and Wind Magnitude, 500mb Temperature, Turbulent Kinetic Energy (TKE), Skew-T, Richardson Number, 850-500mb Vertical Wind Shear, and Base Reflectivity.

## Before Getting Started

Use the package manager [pip]((https://pip.pypa.io/en/stable/)) to install the following packages:

```bash
pip install numpy
pip install matplotlib
pip install cartopy
pip install pygrib
pip install metpy
pip install pandas
pip install sys
pip install PIL
pip install glob
```
## Section 1: Geopotential Height and Wind
### CONUS MAP
1. Download the necessary forecast data file from the HRRR link above (hrrr.t00z.wrfprsf00.grib2) or another date/time if wanted
2. Run the first section of the script labeled Map 1
   - The script will pull the needed values, calculate wind magnitude, and plot a base map with overlayed filled contour for wind speeds in knots, wind barbs, isoheight lines, and a point location of the turbulence.
```bash
map=plt.contourf(lons,lats,mag*1.944,bounds, cmap=plt.cm.afmhot_r,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('knots')
h=plt.contour (lons, lats, hgt, np.arange(np.min(hgt), np.max(hgt),40), linestyles='-', linewidths=2, colors='black', transform=ccrs.PlateCarree())
plt.barbs(lons[::80,::80],lats[::80,::80],uw[::80,::80],vw[::80,::80],transform=ccrs.PlateCarree())
plt.plot(-89.45,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())
```
3. This piece of the script will generate a plot of 500 mb Geopotential Heights and Winds over the Continental United States (CONUS) and be saved as '3.02.HeightWindCONUS.png'.
### REGIONAL MAP
1. Using the same data file as used in Map 1 follow the second section of code titled Map 2
2. Map 2 is the same script as Map 1 with varying central coordinates and extent
```bash
proj=ccrs.LambertConformal(central_longitude=-89.40,central_latitude=35.06)
ax.set_extent([-92.40,-86.40,38.06,32.06])
```
4. This piece will generate the same plot as the one above but it will be zoomed in to view the region in question and be saved as '3.02.HeightWindRegion.png'

### Note
- The script assumes that the forecast data file is located in the same directory as the script so make sure to adjust the file path accordingly
- Other model data files can be used in this script as long as the file is specified

## Section 2: 500 mb Temperature
### CONUS MAP
1. Using the same data from Section 1 select the temperature data for 50000 hPa and convert the data from Kelvin to Celsius
```bash
tmpC = tmp-273.15
```
2. Run the portion of the script labeled Map 3 to plot 500 mb Temperature in a filled contour over CONUS and save it as ('3.02.TemperatureCONUS.png')
```bash
map=plt.contourf(lons,lats,tmpC,bounds, cmap=plt.cm.turbo,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('Celsius')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())
```
### REGIONAL MAP
1. Using the same data file as used in Map 3 follow the second section of code titled Map 4
2. Map 4 is the same script as Map 3 with varying central coordinates and extent
```bash
proj=ccrs.LambertConformal(central_longitude=-89.40,central_latitude=35.06)
ax.set_extent([-92.40,-86.40,38.06,32.06])
```
3. This piece will generate the same plot as the one above but it will be zoomed in to view the region in question and be saved as '3.02.TemperatueRegion.png'

## Section 3: Turbulent Kinetic Energy (TKE)
1. Using the same data as Sections 1 and 2 run the portion of the script labeled as Map 5
2. This script will calculate TKE from the u, v, and w wind components and then reshape the 1D calculated array into a 2D array using the latitude and longitude as the shape base
```bash
from metpy.calc import tke
from metpy.units import units
TKE = metpy.calc.tke(uw, vw, ww, perturbation=False, axis=-1)
TKE2d = np.broadcast_to(TKE[:, np.newaxis], lats.shape)
```
3. From this calculation a plot of one-hour averaged TKE can be created and saved in a file called ('3.02.TKERegion.png')
```bash
map=plt.contourf(lons,lats,TKE2d,np.arange(np.min(TKE),np.max(TKE),.10), cmap=plt.cm.gist_ncar,transform=ccrs.PlateCarree())
cbar = plt.colorbar (location='bottom')
cbar.set_label ('TKE (m2/s2)')
plt.plot(-89.40,35.06, marker='*', color='white', markeredgecolor='black', linewidth=4, markersize = 15, transform = ccrs.PlateCarree())
```
## Section 4: Skew T
### Before Running the Script
- Ensure all packages are installed and imported
```bash
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
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from metpy.plots import add_metpy_logo
```
1. The script is going to utilize data siphoned from the Wyoming Upper-Air archives and is chosen by inputting a specific date, time, and station
2. Run the script labeled as Map 6
   - This portion of the script is going to first calculate all of the thermodynamics needed to plot a Skew T and then plot the figure
3. The last portion of the script will create a hodograph derived from the information calculated above and then save the entire figure as ('3.2.23.SkewT.png')
```bash
agl = sounding['height'] - sounding['height'][0]
mask = agl <= 10 * units.km
intervals = np.array([0, 1, 3, 5, 8]) * units.km
colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:olive']
ax = fig.add_subplot(gs[0, -1])
h = Hodograph(ax, component_range=60.)
h.add_grid(increment=20)
h.plot_colormapped(sounding['u_wind'][mask], sounding['v_wind'][mask], agl[mask], intervals=intervals, colors=colors)
```
## Section 5: Richardson Number
1. Using the same data as the previous sections this script will pull the values for the U and V wind components, Potential Temperature, and Geopotential Height
2. From the pulled values the Richardson Number can be calculated and the data can be cleaned and filtered in accordance with the latitude and longitude
```bash
rn = metpy.calc.gradient_richardson_number(hgt * units('m'), pt * units('K'), uw * units('m/s'), vw * units('m/s'), vertical_dim=0)
invalid_indices = np.isnan(rn)
rn_clean = np.ma.masked_invalid(rn)
southeast = (lons > -92.0) & (lons < -86.0)
filtered_lons = lons[southeast]
filtered_rn_clean = rn_clean[southeast]
rn_range_indices = (filtered_rn_clean >= -20) & (filtered_rn_clean <= 10)
filtered_lons_range = filtered_lons[rn_range_indices]
filtered_rn_clean_range = filtered_rn_clean[rn_range_indices]
```
3. The values from this calculation are then plotted spatially over the region in question and saved as ('3.02.RichardsonNumberGraph.png')

## Section 6: 850-500 mb Vertical Wind Shear 
### Both CONUS and REGIONAL Maps
1. the script will calculate vertical wind shear hourly over a five-hour time period
   - Ensure that all data files are downloaded and file paths are adjusted as needed
```bash
data2 = pygrib.open ('hrrr.t23z.wrfprsf00.grib2')
data3 = pygrib.open ('hrrr.t22z.wrfprsf00.grib2')
data4 = pygrib.open ('hrrr.t21z.wrfprsf00.grib2')
data5 = pygrib.open ('hrrr.t20z.wrfprsf00.grib2')
```
2. Run the entire Wind Shear section labeled Map 9 of the script in order to ensure that no variables get mixed up and the figures plot correctly
3. The values calculated will be plotted as filled contours and the output will be 10 separate PNG images that can be used to create a time series loop
### Note
- If this portion of the script runs out of order be sure to re-open the file in which you need or the variables could potentially get mixed up

## Section 7: Radar Base Reflectivity
1. Ensure all NEXRAD Level II data files are downloaded and are available in the same directory as the script 
   - The files include (KNQA20230301_230141_V06, KNQA20230301_230715_V06, etc.) and there are 20 total files used in this script
2. Import all necessary packages
```bash
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from metpy.calc import azimuth_range_to_lat_lon
from metpy.cbook import get_test_data
from metpy.io import Level2File
from metpy.plots import add_metpy_logo, add_timestamp, USCOUNTIES
from metpy.units import units
```
3. Run each portion of the script separately to ensure each file uploads correctly and to make any adjustments to the title or save file name if using different dates/times
4. The script will plot 20 separate base reflectivity images with the point location clearly marked
   - The created PNG files can be put into a loop to show a time series of radar
```bash
import sys, os
from PIL import Image
import glob

fGIF = "Radar.gif"
H = 1152
W = 1152
n = 1
frames = []
images = glob.glob("Radar*.png")
#resize the images 
for i in images:
    newImg = Image.open(i)
    if (len(sys.argv) < 2 and n > 0):
        newImg = newImg.resize((W, H))
    frames.append(newImg)
frames[0].save(fGIF, format='GIF', append_images=frames[1:],
    save_all=True, duration=700, loop=0)
```
