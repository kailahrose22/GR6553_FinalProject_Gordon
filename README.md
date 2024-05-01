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
