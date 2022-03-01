#! /usr/bin/env python

# This script prepares ESM output with LCLM chessboard pattern 
# to separate into local, nonlocal, and total effects 

# It is based on the scripts of Johannes Winckler for his 2017 study
# doi: 10.1175/JCLI-D-16-0067.1


# This function takes a difference map the variable name and the interpolation method.

# Calculate everything on extended grid, such that values around zero meridian can be interpolated correctly.
# Output are interpolated local, non-local, and total effects.

# run the script as: python prepare_local_nonlocal_total_effects.py FILE_DIFFERENCE_MAP VARIABLE_NAME INTERPOLATION_METHOD

import numpy as np
#from scipy.interpolate import RectBivariateSpline
import scipy.interpolate as interpolate
import datetime as dt
import os
from copy import deepcopy as cp
import netCDF4 as nc
import sys
from dask import delayed
import pandas as pd
import xarray as xr
import time
from collections import Counter
import matplotlib.pyplot as plt
import pickle

# INTERPOLATION METHOD = "interpolate_total" or "interpolate_local"

# GET INFO FROM INPUT
if os.path.isfile(sys.argv[1]):
    FILE_DIFFERENCEMAP = sys.argv[1]
    if "interpolate" in sys.argv[2]:
        INTERPOLATE_METHOD = sys.argv[2]
        VAR = sys.argv[3]
    else:
        INTERPOLATE_METHOD = sys.argv[3]
        VAR = sys.argv[2]
    
elif os.path.isfile(sys.argv[2]):    
    FILE_DIFFERENCEMAP = sys.argv[2]
    if "interpolate" in sys.argv[1]:
        INTERPOLATE_METHOD = sys.argv[1]
        VAR = sys.argv[3]
    else:
        INTERPOLATE_METHOD = sys.argv[3]
        VAR = sys.argv[1]
    
    
elif os.path.isfile(sys.argv[3]):    
    FILE_DIFFERENCEMAP = sys.argv[3]
    if "interpolate" in sys.argv[2]:
        INTERPOLATE_METHOD = sys.argv[2]
        VAR = sys.argv[1]
    else:
        INTERPOLATE_METHOD = sys.argv[1]
        VAR = sys.argv[2]
            
else:    
    print(sys.argv[:], " there is no file or wrong interpolation method in the parsed arguments...")
    sys.exit( 2)

# GET INFO ABOUT THE MODEL
if "cesm" in FILE_DIFFERENCEMAP:
    MODEL = "cesm"
    FILL_VALUE = 1e36
elif "mpiesm" in FILE_DIFFERENCEMAP:
    MODEL = "mpiesm"
    FILL_VALUE = 1e20
elif "ecearth" in FILE_DIFFERENCEMAP:
    MODEL = "ecearth"
    FILL_VALUE=9.96921e36        
else:
    print("The model could not be identified... :|")
    sys.exit( 2)
    

#INPUT_WD = "/home/felixh/data/mpiesm/chessboard/"
INPUT_WD = "/pf/b/b380948/lamaclima/chessboard_tests/"
DATA_WD = "/home/felixh/data/mpiesm/differencemaps/ctl-crop/"

# IMPORT GLOBAL CHESSBOARD DISTRIBUTION ON LAND
#ifile = INPUT_WD + "lamaclima_experiments_chessboard_pattern_no-missing.nc"
if MODEL != "ecearth":   ##data with a regular grid
    ifile = INPUT_WD + "lamaclima_experiments_chessboard_pattern_" + MODEL + ".nc"
    f = nc.Dataset(ifile, 'r')
    CHESSBOARD = f.variables['chessboard_pattern'][:]
    LAT = f.variables['lat'][:]
    LON = f.variables['lon'][:]
    f.close()
    # IMPORT GLOBAL CHESSBOARD DISTRIBUTION GLOBALLY
    ifile = INPUT_WD + "lamaclima_experiments_global_chessboard_pattern_" + MODEL + ".nc"
    f = nc.Dataset(ifile, 'r')
    GLOBAL_CHESSBOARD = f.variables['chessboard_pattern'][:]
    f.close()

else:
    CHESSBOARD_LAT=pd.read_csv('/pf/b/b381334/files_ec_earth/Chessboard_normal_ctrl_grids_info/EC-Earth_CTRL_Map_Lat.csv')
    CHESSBOARD_LON=pd.read_csv('/pf/b/b381334/files_ec_earth/Chessboard_normal_ctrl_grids_info/EC-Earth_CTRL_Map_Lon.csv')
    CHESSBOARD_LAT=CHESSBOARD_LAT.iloc[:,1].values
    CHESSBOARD_LON=CHESSBOARD_LON.iloc[:,1].values
    
    ifile = "/pf/b/b381334/files_ec_earth/ec-earth_land_sea_mask.nc"
    f = nc.Dataset(ifile, 'r')
    #SEA_LAND_MASK = f.variables['LANDFRAC_PFT'][:]
    SEA_LAND_MASK = f.variables['var172'][:] * 1.0
    GLACIER = f.variables['var172'][:] * 0.0  # no glacier mask provided for ecearth so I just give the land sea mask again
    #CHECKERS=np.ones(len(LAT))
    #print(len(LAT))
    #print(len(CHESSBOARD_LAT))
    #for i in range(len(LAT)):
    #    for j in range(len(CHESSBOARD_LAT)):
            #print([np.round(LAT[i],3),np.round(LON[i],3)],[np.round(CHESSBOARD_LAT[j],3),np.round(CHESSBOARD_LON[j],3)])
    #        if [np.round(LAT[i],3),np.round(LON[i],3)]==[np.round(CHESSBOARD_LAT[j],3),np.round(CHESSBOARD_LON[j],3)]:
    #           CHECKERS[i]=1
               #print('chess')
    #           i=i+1
    #        else:
    #           CHECKERS[i]=0
            #time.sleep(0.1)
    #print(CHECKERS)
    #da= xr.DataArray(data=CHECKERS,dims=["space"],coords=dict(space=(["space"], LAT)),attrs=dict(description="Checkerboard pattern EC-EARTH"))
    #da.to_netcdf("/scratch/b/b381334/files_ec_earth/Chessboard_normal_ctrl_grids_info/lamaclima_experiments_global_chessboard_pattern_ecearth.nc")
    # IMPORT CHESSBOARD DISTRIBUTION (only over land for ecearth)
    ifile = "/pf/b/b381334/files_ec_earth/Chessboard_normal_ctrl_grids_info/lamaclima_experiments_global_chessboard_pattern_ecearth.nc"
    f = nc.Dataset(ifile, 'r')
    CHESSBOARD = f.variables['__xarray_dataarray_variable__'][:]

    #read in grid data (only lat defined, lon needs to be deduced based on spatial resolution definition
    grid=pd.read_csv('/pf/b/b381334/files_ec_earth/Chessboard_normal_ctrl_grids_info/ec_earth_grid_info.csv', thousands=',')
    lat_grid=grid.iloc[3:,3].str.replace(',', '').astype(float).div(1000000).reset_index(drop=True).values
    rpoints=grid.iloc[3:,1]
    delta_lon=rpoints.astype(float).div(360)
    delta_lon=delta_lon.pow(-1).reset_index(drop=True).values
    ## make global LAT and LON arrays
    tel_lat=0
    tel=1
    LAT=np.zeros((len(SEA_LAND_MASK[0,:]),1))
    LON=np.zeros((len(SEA_LAND_MASK[0,:]),1))
    tel_lat_old=20 #value gets overwritten during loop just placeholder
    for i in range(len(SEA_LAND_MASK[0,:])):
        tel=tel+1
        delta=delta_lon[[tel_lat]]
        LAT[i]= lat_grid[[tel_lat]]
        if (LAT[i]<1 and LAT[i]>-1):
            LAT[i]=LAT[i]*1000000    #these have been read in wrong so need to be corrected
        #time.sleep(0.5)
        if tel_lat !=tel_lat_old:  ##check if first instance of lat band, start at lon=0
            LON[i]=0  
        else:
            LON[i]=LON[i-1]+delta
        tel_lat_old=tel_lat
        if int(360/delta)==int(tel-1):
            tel_lat=tel_lat+1
            tel=1
    ##note: I checked this calculation against the real lat and lon and it checks out
        
    for i in range(len(SEA_LAND_MASK[0,:])):  
        if LON[i]>180:
            LON[i]=LON[i]-360 #make lon range from -180 to 180
    #f.close()
    #saving these files so I can access them more quickly when needed
    pickle.dump(LAT, open('/pf/b/b381334/lat_ec_earth.sav', 'wb'))
    pickle.dump(LON, open('/pf/b/b381334/lon_ec_earth.sav', 'wb'))
    pickle.dump(CHESSBOARD_LAT, open('/pf/b/b381334/checkers_lat_ec_earth.sav', 'wb'))
    pickle.dump(CHESSBOARD_LON, open('/pf/b/b381334/checkers_lon_ec_earth.sav', 'wb'))

 
# IMPORT SEA LAND MASK (except EC-EARTH)
if MODEL == "mpiesm":
    ifile = INPUT_WD + "jsbach_T63GR15_11tiles_5layers_2005_dynveg_slm_glac.nc"
    f = nc.Dataset(ifile, 'r')
    SEA_LAND_MASK = f.variables['slm'][:]
    GLACIER = f.variables['glac'][:]    
    f.close()    


elif MODEL == "cesm":
    ifile = INPUT_WD + "landmask_glacier_cesm.nc"
    f = nc.Dataset(ifile, 'r')
    #SEA_LAND_MASK = f.variables['LANDFRAC_PFT'][:]
    SEA_LAND_MASK = f.variables['landmask'][:] * 1.0
    GLACIER = f.variables['GLACIER_REGION'][:] * 1.0 # simple transformation to float...
    f.close()    



# IMPORT DIFFERENCEMAP
#ifile = DATA_WD + "nep_Emon_MPI-ESM1-2-LR_215501-217412_ctl-crop_timavg.nc"
ifile = FILE_DIFFERENCEMAP
f = nc.Dataset(ifile, 'r+')
# CHECK IF INVERTED LAT VALUES IN ORIGINAL FILE AND KEEP THE ORIGINAL CONVENTION

if MODEL != 'ecearth':
   LAT_DIFF = f.variables["lat"][:]
   LON_DIFF = f.variables["lon"][:]
else:
   LAT_DIFF=LAT[:]
   LON_DIFF=LON[:]

if np.around( LAT_DIFF[0], 5) == np.around( LAT[0], 5):
    i = 0
    j = 1    

elif np.around( LAT_DIFF[0], 5) == np.around( -1 * LAT[0], 5):
    i = -1
    j = -1
    
elif VAR=='irrigation_flux':
    i=0
    j=1

else:
    print("Latitude information of used masks doesn't match... :|")
    sys.exit( 2)
##ecearth data does not have units defined
if MODEL=='ecearth' or VAR=='albedo' or VAR=='TS_ebal' or VAR=='irrigation_flux' or VAR=='ts_ebal' or VAR=='wbgtid' or VAR=='wbgtid_iso_400W' or VAR=='SW_frac' or VAR=='LW_frac' or VAR=='avE' or VAR=='ts_ebal_x' or VAR=='ts_ebal_n' or VAR=='ts_ebal_meandaymax' or VAR=='ts_ebal_meandaymin' or VAR=='DTR' or VAR=='wbgtod' or VAR=='wbgtod_iso_400W':
	UNIT='/'
else:
	UNIT = f.variables[VAR].units  
NLAT = len(LAT_DIFF)
NLON = len(LON_DIFF)
#DIFFERENCEMAP_ALLTIMESTEPS = f.variables[VAR][:,-1::-1,:] # time, lat, lon
if MODEL!='ecearth':
   DIFFERENCEMAP_ALLTIMESTEPS = f.variables[VAR][:,i::j,:] # time, lat, lon
   NSTEPS = len( DIFFERENCEMAP_ALLTIMESTEPS[:,0,0])
elif MODEL=='ecearth':
   DIFFERENCEMAP_ALLTIMESTEPS = f.variables[VAR][:,i::j] # time, rgrid
   NSTEPS = len( DIFFERENCEMAP_ALLTIMESTEPS[:,0])
   if DIFFERENCEMAP_ALLTIMESTEPS.ndim>2:                 #happens for variable with vertical layer (eg soil temperature) here we take top layer
       DIFFERENCEMAP_ALLTIMESTEPS=DIFFERENCEMAP_ALLTIMESTEPS[:,0,:]
#f.close()

# ----- CREATE CONSTANT INPUT MASKS

# 4 % EXTENDED NUMBER OF GRIDPOINTS IN LON DIRECTION NEEDED FOR INTERPOLATION AROUND 0 DEGREE W
# WILL BE RESTRICTED TO ORIGINAL GRID LATER.

# FOR MPI-ESM: INSTEAD OF 192 LONGITUDE ROWS WE ADD 8 LONGITUDE ROWS AND USE 200 LONGITUDE ROWS
#EXTEND_LON_NUMBER = int( np.around( len(LON) * 1.04, 0))
EXTEND_LON_NUMBER = len(LON) + 8

## CREATE EXTENDED CHESSBOARD MAP ON ALL LAND AND GLOBAL GRID POINTS -- non masked arrays
#CHESSBOARD_EXTENDED = np.tile( CHESSBOARD,2)[:,0:EXTEND_LON_NUMBER]
#GLOBAL_CHESSBOARD_EXTENDED = np.tile( GLOBAL_CHESSBOARD,2)[:,0:EXTEND_LON_NUMBER]

# CREATE EXTENDED CHESSBOARD MAP ON ALL LAND AND GLOBAL GRID POINTS -- masked arrays
if MODEL!='ecearth':  ##currently not done for ecearth as checkers are lkand only, only location where issue may arise is siberia alaskan border
   CHESSBOARD_EXTENDED = np.ma.masked_equal( np.tile( CHESSBOARD,2)[:,0:EXTEND_LON_NUMBER], value=FILL_VALUE)
   GLOBAL_CHESSBOARD_EXTENDED = np.ma.masked_equal( np.tile( GLOBAL_CHESSBOARD,2)[:,0:EXTEND_LON_NUMBER], value=FILL_VALUE)
   # CREATE EXTENDED SEA LAND MASK AND GLACIER MASK -- masked arrays
   LAND_EXTENDED = np.ma.masked_equal( np.tile( SEA_LAND_MASK,2)[:,0:EXTEND_LON_NUMBER], value=FILL_VALUE)
   GLACIER_EXTENDED = np.ma.masked_equal( np.tile( GLACIER, 2)[:,0:EXTEND_LON_NUMBER], value=FILL_VALUE)  
   # REVERSE GLOBAL CHESSBOARD GRID POINTS (1-->0; 0-->1)
   GLOBAL_CHESSBOARD_EXTENDED_REVERSE = np.abs( GLOBAL_CHESSBOARD_EXTENDED - 1)
#else:
    ## TO DO add more than one value per lat band (normally should be 8) and need to add these at different spatial coordinates
  # index_list=[]
   #lat_list=[]
   #for i in range(len(LAT)):
   #    current_lat=LAT[i]
   #    if lat_list.count(current_lat)==0:
   #        lat_list.append(current_lat)
   #        index_list_lat= (LAT==current_lat).nonzero()
   #        min_lon=np.min(LON[index_list_lat])
   #        index_list_lon= (LON==min_lon).nonzero()
   #        j=0
   #        final_index=np.intersect1d(index_list_lat,index_list_lon)
   #        index_list.append(int(final_index[:]))
   #LAT_EXTENDED=np.zeros(len(LAT)+len(index_list))
   #LAT_EXTENDED[:len(LAT)]=LAT[:]
   #LAT_EXTENDED[len(LAT):]=LAT[index_list]
   #print(LAT_EXTENDED[len(LAT):])
   #LON_EXTENDED=np.zeros(len(LON)+len(index_list))
   #LON_EXTENDED[:len(LON)]=LON[:]
   #LON_EXTENDED[len(LON):]=LON[index_list]-1
   #print(LON_EXTENDED[len(LON):])
   #CHESSBOARD_EXTENDED=np.zeros(len(CHESSBOARD)+len(index_list))
   #CHESSBOARD_EXTENDED[:len(CHESSBOARD)]=CHESSBOARD[:]
   #CHESSBOARD_EXTENDED[len(CHESSBOARD):]=CHESSBOARD[index_list]
   #GLOBAL_CHESSBOARD_EXTENDED=np.zeros(len(GLOBAL_CHESSBOARD)+len(index_list))
   #GLOBAL_CHESSBOARD_EXTENDED[:len(GLOBAL_CHESSBOARD)]=GLOBAL_CHESSBOARD[:]
   #GLOBAL_CHESSBOARD_EXTENDED[len(GLOBAL_CHESSBOARD):]=GLOBAL_CHESSBOARD[index_list]

## CREATE EXTENDED SEA LAND MASK AND GLACIER MASK -- non masked arrays
#LAND_EXTENDED = np.tile( SEA_LAND_MASK,2)[:,0:EXTEND_LON_NUMBER]
#GLACIER_EXTENDED = np.tile( GLACIER,2)[:,0:EXTEND_LON_NUMBER]

def interpolate_nonlocal( chessboard_extended, differencemap_extended, land_extended, coastboxes):
    
    #---------coordinate of the grid boxes that are interpolated
    points = np.where( np.all( np.array(( chessboard_extended == 0, land_extended == 1)), axis = 0))
    values = differencemap_extended[np.all( np.array(( chessboard_extended == 0, land_extended == 1)), axis = 0)]
    #values = differencemap_extended[points]
    
    #grid_x, grid_y = np.mgrid[0:96:1, 0:land_extended.shape[1]:1]
    grid_x, grid_y = np.mgrid[0:land_extended.shape[0]:1, 0:land_extended.shape[1]:1]
    
    interped = interpolate.griddata( np.array(( points[1] * 1.0, points[0] * 1.0)).T, values,\
        (grid_y,grid_x), method='linear')
    interped_nearest = interpolate.griddata( np.array(( points[1] * 1.0, points[0] * 1.0)).T, values,\
        (grid_y,grid_x), method='nearest')
    
    non_local = cp( interped)
    non_local[coastboxes == 1] = interped_nearest[coastboxes == 1]
    # take values over the ocean directly
    non_local[land_extended == 0] = differencemap_extended[land_extended == 0]

    return non_local

def interpolate_total( chessboard_extended, differencemap_extended, land_extended, glacier_extended, coastboxes):
    
    #---------coordinate of the grid boxes that are interpolated
    points = np.where( chessboard_extended == 1)
    values = differencemap_extended[ chessboard_extended == 1]
    
    #grid_x, grid_y = np.mgrid[0:96:1, 0:land_extended.shape[1]:1]
    grid_x, grid_y = np.mgrid[0:land_extended.shape[0]:1, 0:land_extended.shape[1]:1]
    
    interped = interpolate.griddata( np.array(( points[1] * 1.0, points[0] * 1.0)).T, values,\
        (grid_y,grid_x), method='linear')
    interped_nearest = interpolate.griddata( np.array(( points[1] * 1.0, points[0] * 1.0)).T, values,\
        (grid_y,grid_x), method='nearest')
    
    total_timeseries_interpolated = cp( interped)
    # interpolate values at the coast differently
    total_timeseries_interpolated[coastboxes == 1] = interped_nearest[coastboxes == 1]
    # take values over the ocean directly
    total_timeseries_interpolated[land_extended == 0] = differencemap_extended[land_extended == 0]
    # take values over the glaciers directly --> this happens also implicitly for nonlocal interpolation
    total_timeseries_interpolated[glacier_extended == 1] = differencemap_extended[glacier_extended == 1]

    return total_timeseries_interpolated


def interpolate_local( chessboard_extended, local_extended, land_extended, glacier_extended, coastboxes):
    
    #---------coordinate of the grid boxes that are interpolated
    points = np.where( chessboard_extended == 1)
    values = local_extended[ chessboard_extended == 1]
    
    #grid_x, grid_y = np.mgrid[0:len(LAT):1, 0:land_extended.shape[1]:1]
    grid_x, grid_y = np.mgrid[0:land_extended.shape[0]:1, 0:land_extended.shape[1]:1]
    
    interped = interpolate.griddata( np.array(( points[1] * 1.0, points[0] * 1.0)).T, values,\
        (grid_y,grid_x), method='linear')
    interped_nearest = interpolate.griddata( np.array(( points[1] * 1.0, points[0] * 1.0)).T, values,\
        (grid_y,grid_x), method='nearest')
        
    local_interp = cp( interped)
    local_interp[coastboxes == 1] = interped_nearest[coastboxes == 1]
    
    # MASK OUT WATER BODIES AND GLACIER TOGETHER
    local_interp[land_extended == 0] = 0
    local_interp[glacier_extended == 1] = 0

    return local_interp

def subtract(extended_diff_map, remote_extended):
    return extended_diff_map - remote_extended

    
def determine_coast_boxes(chessboard_extended, global_chessboard_extended, global_chessboard_extended_reverse):
    # coastpixels are boxes that are ocean or that are near the ocean,
    # such that an ocean grid box would be used for interpolating the lcc grid boxes
    coastboxes = np.zeros( chessboard_extended.shape)
    #coastboxes_2 = np.zeros( chessboard_extended.shape)
    
    for latindex in range( chessboard_extended.shape[0]):
        for lonindex in range(chessboard_extended.shape[1]):
            # in a 9 boxes window around the grid box, look if there are more global chessboard pixels
            # than land chessboard pixels
            # Johannes took global_chessboard_extended_reverse for his calculation....
            if (np.sum( global_chessboard_extended[latindex-2:latindex+3,lonindex-2:lonindex+3]) - np.sum( chessboard_extended[latindex-2:latindex+3,lonindex-2:lonindex+3]) > 0):
                coastboxes[latindex,lonindex]=1
            #if (np.sum( global_chessboard_extended_reverse[latindex-2:latindex+3,lonindex-2:lonindex+3]) -
                #np.sum( chessboard_extended[latindex-2:latindex+3,lonindex-2:lonindex+3]) > 0):
                #coastboxes_2[latindex,lonindex]=1            
    return coastboxes#, coastboxes_2



def write_netcdf( total_timeseries, non_local_timeseries,local_timeseries, f, UNIT, LAT_DIFF, NSTEPS, NLAT, NLON, INTERPOLATE_METHOD):
#def write_netcdf( total_timeseries, non_local_timeseries, f, UNIT, LAT_DIFF, NSTEPS, NLAT, NLON):
    # USE COPY OF FILE TO CREATE NEW NETCDF FILE
    print("Writing netCDF4 file...")
    #ifile = FILE_DIFFERENCEMAP.replace(".nc","") + "_signal-separated.nc"
    #f = nc.Dataset( ifile, 'r+') # format='NETCDF4')

    # Define global attributes
    f.creation_date=str(dt.datetime.today())
    f.contact='felix.havermann@lmu.de; Ludwig-Maximilians-University Munich'
    f.comment='Produced with Script ' + os.path.basename(__file__)
    #f.title='Land cover fractions of MPIESM-1.2-LR LAMACLIMA ' + SIMULATION_TYPE + ' simulation'
    #f.subtitle=''
    #f.anything=''
    #f_restart.comment='Only the variable cover_fract_pot was changed to a 100 % forest world and all other cover types are set to zero.\n'\
    #f.comment='Only the variable cover_fract_pot was changed to a 100 % forest world and all other cover types are set to fract_small (1e-10).\n'\
        #'This file is used for a 100 % forest simulation within the LAMACLIMA project\n'\
        #'Produced with Script ' + os.path.basename(__file__)
    #f.createVariable(LCT_CONVERSION[key][1], np.float32,('lat','lon'), fill_value=FILL_VALUE)

    # NON-LOCAL SIGNAL 
    f.createVariable( VAR + "_nonlocal", np.float64,('time','lat','lon'), fill_value=FILL_VALUE)
    #f.variables[VAR + "_nonlocal"][0,:,:] = non_local_timeseries[:,i::j,:]
    #f.variables[VAR + "_nonlocal"][1:,:,:] = np.zeros(( NSTEPS-1, NLAT, NLON))
    f.variables[VAR + "_nonlocal"][:,:,:] = non_local_timeseries[:,i::j,:]
    f.variables[VAR + "_nonlocal"].longname = 'Non-local (=remote) effect of LCLM change'
    f.variables[VAR + "_nonlocal"].units = UNIT
    f.variables[VAR + "_nonlocal"].grid_type = 'gaussian'

    if INTERPOLATE_METHOD == "interpolate_local":
        # TOTAL SIGNAL
        total_timeseries_summed_up = local_timeseries[:,i::j,:] + non_local_timeseries[:,i::j,:]
        f.createVariable( VAR + "_total", np.float64,('time','lat','lon'), fill_value=FILL_VALUE)
        #f.variables[VAR + "_total"][0,:,:] = local_interpolated[:,i::j,:] + non_local_interpolated[:,i::j,:]
        f.variables[VAR + "_total"][:,:,:] = total_timeseries_summed_up
        #f.variables[VAR + "_total"][1:,:,:] = np.zeros(( NSTEPS-1, NLAT, NLON))
        f.variables[VAR + "_total"].longname = 'Total (=local + non-local) effect of LCLM change'
        f.variables[VAR + "_total"].units = UNIT
        f.variables[VAR + "_total"].grid_type = 'gaussian'
        
        # LOCAL SIGNAL
        MASK = np.tile( CHESSBOARD[i::j,:], (NSTEPS,1,1))
        f.createVariable( VAR + "_local", np.float64,('time','lat','lon'), fill_value=FILL_VALUE)
        #f.variables[VAR + "_local"][0,:,:] = local_interped[:,i::j,:]
        #f.variables[VAR + "_local"][0,:,:] = local_timeseries[i::j,:]
        #f.variables[VAR + "_local"][1:,:,:] = np.zeros(( NSTEPS-1, NLAT, NLON))
        f.variables[VAR + "_local"][:,:,:] = np.ma.masked_array( local_timeseries[:,i::j,:], MASK.mask)
        f.variables[VAR + "_local"].longname = 'Local effect of LCLM change'
        f.variables[VAR + "_local"].units = UNIT
        f.variables[VAR + "_local"].grid_type = 'gaussian'    
    
    
    elif INTERPOLATE_METHOD == "interpolate_total":
        # TOTAL SIGNAL
        f.createVariable( VAR + "_total", np.float64,('time','lat','lon'), fill_value=FILL_VALUE)
        f.variables[VAR + "_total"][:,:,:] = total_timeseries[:,i::j,:]
        f.variables[VAR + "_total"].longname = 'Total effect of LCLM change'
        f.variables[VAR + "_total"].units = UNIT
        f.variables[VAR + "_total"].grid_type = 'gaussian'
    
        # LOCAL SIGNAL
        local_timeseries_difference = total_timeseries[:,i::j,:] - non_local_timeseries[:,i::j,:]
        
        GLACIER_MASK = np.tile( GLACIER[i::j,:], (NSTEPS,1,1))
        local_timeseries_difference[GLACIER_MASK==1] = 0
        
        MASK = np.tile( CHESSBOARD[i::j,:], (NSTEPS,1,1))
        #MASK = np.tile( CHESSBOARD, (NSTEPS,1,1))
        f.createVariable( VAR + "_local", np.float64,('time','lat','lon'), fill_value=FILL_VALUE)
        f.variables[VAR + "_local"][:,:,:] = np.ma.masked_array( local_timeseries_difference, MASK.mask)
        f.variables[VAR + "_local"].longname = 'Local effect of LCLM change'
        f.variables[VAR + "_local"].units = UNIT
        f.variables[VAR + "_local"].grid_type = 'gaussian'


    f.close()
    print("writing finished.")


def get_original_grid(grid_extended, longitude, time_step, timeseries):
    tmp1 = grid_extended[:, 4:len(longitude)]    
    tmp2 = grid_extended[:, len(longitude):len(longitude)+4]
    timeseries[time_step,:,:] = np.hstack((tmp2, tmp1))
    
    return timeseries

#def get_original_grid_nonlocal(grid_extended, LON, step, non_local_timeseries):
    #tmp1 = grid_extended[:, 4:len(LON)]    
    #tmp2 = grid_extended[:, len(LON):len(LON)+4]
    #non_local_timeseries[step,:,:] = np.hstack((tmp2, tmp1))
    
    #return non_local_timeseries

#def get_original_grid_local(grid_extended, LON, step, local_timeseries):
    #tmp1 = grid_extended[:, 4:len(LON)]
    #tmp2 = grid_extended[:, len(LON):len(LON)+4]
    #local_timeseries[step,:,:] = np.hstack((tmp2, tmp1))
    
    #return local_timeseries

# DETERMINE COASTBOXES TO APPLY NEAREST NEIGHBOUR CALCULATION FOR THOSE BOXES      
if MODEL !='ecearth':
    coastboxes = determine_coast_boxes( CHESSBOARD_EXTENDED, GLOBAL_CHESSBOARD_EXTENDED, GLOBAL_CHESSBOARD_EXTENDED_REVERSE)

    non_local_timeseries = np.ma.masked_all((NSTEPS, NLAT, NLON))
    local_timeseries = np.ma.masked_all((NSTEPS, NLAT, NLON))
    total_timeseries = np.ma.masked_all((NSTEPS, NLAT, NLON))
else:   #can't use functions created above for ec-earth
    ##determine coastboxes
    coastboxes = np.zeros( LAT.shape)
    coastboxes_checkers = np.zeros( CHESSBOARD_LAT.shape)
    #coastboxes_2 = np.zeros( chessboard_extended.shape)
    #plt.scatter(LON,LAT,c=SEA_LAND_MASK,cmap='hot',s=0.1)
    #plt.colorbar()
    #plt.show()
    if os.path.isfile('/pf/b/b381334/signal_seperation/coast_checkers.sav'):
        coastboxes_checkers = pickle.load(open('/pf/b/b381334/signal_seperation/coast_checkers.sav', 'rb'))
        coastboxes = pickle.load(open('/pf/b/b381334/signal_seperation/coast.sav', 'rb'))
        translation_table = pickle.load(open('/pf/b/b381334/signal_seperation/trans_table.sav', 'rb'))
    else:
        from scipy.spatial import cKDTree
        import pickle
        tree = cKDTree(np.c_[LAT,LON])
        dd, ii = tree.query([0,0], k=8)
        translation_table=np.zeros((len(CHESSBOARD_LAT),1))  ##array to transfer from checkersboards land only grid to the full global grid
        translation_table_missing=np.zeros((len(CHESSBOARD_LAT),1))   ## array to check for values not matching between land only and global grids
        for i in range(len(CHESSBOARD_LAT)):
            if np.any(np.logical_and(np.round(LAT,1)==np.round(CHESSBOARD_LAT[i],1),np.round(LON,1)==np.round(CHESSBOARD_LON[i],1))) !=True:
                #print('shit')
                pass
                translation_table_missing[i,0]=1
            else:
            #print(np.logical_and(np.round(LAT,3)==np.round(CHESSBOARD_LAT[i],3),np.round(LON,3)==np.round(CHESSBOARD_LON[i],3)))
                #print(np.where(np.logical_and(np.round(LAT,3)==np.round(CHESSBOARD_LAT[i],3),np.round(LON,3)==np.round(CHESSBOARD_LON[i],3))))
                translation_table[i,0]=np.where(np.logical_and(np.round(LAT,2)==np.round(CHESSBOARD_LAT[i],2),np.round(LON,2)==np.round(CHESSBOARD_LON[i],2)))[0]
            ## find nearest neighbours of each location on land sea mask
                translation_table=translation_table.astype('int')
                dd, ii = tree.query(np.c_[LAT[translation_table[i,0]],LON[translation_table[i,0]]], k=8)
                # in a 9 boxes window around the grid box, look if there are more global chessboard pixels
                # than land chessboard pixels
                # Johannes took global_chessboard_extended_reverse for his calculation....
                #if (np.sum( SEA_LAND_MASK[0,ii]) > 1 and np.sum( SEA_LAND_MASK[0,ii])<7.5):
                if (np.sum( SEA_LAND_MASK[0,ii]) > 1 and np.sum( SEA_LAND_MASK[0,ii])<7):
                    #print(SEA_LAND_MASK[0,ii])
                    #print(ii)
                    coastboxes[translation_table[i,0]]=1
                    coastboxes_checkers[i]=1
                translation_table_missing[i,0]=np.nan
        
        
        #for i in range( len(LAT)):
            #if np.where(np.logical_and(np.round(CHESSBOARD_LAT,3)==np.round(LAT[i],3),np.round(CHESSBOARD_LON,3)==np.round(LON[i],3))):
                #translation_table[j,0]=i
                #print(j)
                #print(i)
                #print(translation_table[0:5,0])
                #time.sleep(0.5)

                ## find nearest neighbours of each location on land sea mask
                #dd, ii = tree.query(np.c_[LAT[i],LON[i]], k=8)
                # in a 9 boxes window around the grid box, look if there are more global chessboard pixels
                # than land chessboard pixels
                # Johannes took global_chessboard_extended_reverse for his calculation....
                #if (np.sum( SEA_LAND_MASK[0,ii]) > 1 and np.sum( SEA_LAND_MASK[0,ii])<7.5):
                    #print(SEA_LAND_MASK[0,ii])
                    #print(ii)
                    #coastboxes[i]=1
                    #coastboxes_checkers[j]=1
                    
        pickle.dump(coastboxes, open('/pf/b/b381334/signal_seperation/coast.sav', 'wb'))
        pickle.dump(coastboxes_checkers, open('/pf/b/b381334/signal_seperation/coast_checkers.sav', 'wb'))
        pickle.dump(translation_table, open('/pf/b/b381334/signal_seperation/trans_table.sav', 'wb'))
        
##plot to check amount of missing matches, not sure why we have these but seems to be small enough not to cause any real issues, generally located at 0 long in boreal latitudes (both NH and SH)
#plt.scatter(CHESSBOARD_LON,CHESSBOARD_LAT,c=translation_table_missing,cmap='hot',s=0.1)
#plt.colorbar()
#plt.show()
        
if MODEL != 'ecearth':
    for step in range( NSTEPS):
        #print("Step", step, "of")
    
        differencemap = DIFFERENCEMAP_ALLTIMESTEPS[step,:,:]
        ## CREATE EXTENDED DIFFERENCE MAPS FROM CTL, CROP, IRR, FRST, AND HARV EXPERIMENTS -- non masked arrays
        #DIFFERENCEMAP_EXTENDED = np.tile( differencemap,2)[:,0:EXTEND_LON_NUMBER]

        # CREATE EXTENDED DIFFERENCE MAPS FROM CTL, CROP, IRR, FRST, AND HARV EXPERIMENTS -- masked arrays
        DIFFERENCEMAP_EXTENDED = np.ma.masked_equal( np.tile( differencemap, 2)[:,0:EXTEND_LON_NUMBER], value=FILL_VALUE)

        # INTERPOLATE DIFFERENT EFFECTS AND RESTRICT TO ORIGINAL GRID 
        #nonlocal_extended = interpolate_nonlocal( CHESSBOARD_EXTENDED, DIFFERENCEMAP_EXTENDED, LAND_EXTENDED, coastboxes)
        nonlocal_extended = delayed(interpolate_nonlocal)( CHESSBOARD_EXTENDED, DIFFERENCEMAP_EXTENDED, LAND_EXTENDED, coastboxes)
        
        #non_local_timeseries = delayed(get_original_grid_nonlocal)(nonlocal_extended, LON, step, non_local_timeseries)
        non_local_timeseries = delayed(get_original_grid)(nonlocal_extended, LON, step, non_local_timeseries)    
        
        if INTERPOLATE_METHOD == "interpolate_total":
            # CALCULATE TOTAL EFFECTS BY JUST INTERPOLATING GRID POINTS WICH LCLM CHANGE
            total_extended = delayed(interpolate_total)( CHESSBOARD_EXTENDED, DIFFERENCEMAP_EXTENDED, LAND_EXTENDED, GLACIER_EXTENDED, coastboxes)
            
            total_timeseries = delayed(get_original_grid)(total_extended, LON, step, total_timeseries)

        elif INTERPOLATE_METHOD == "interpolate_local":
            #local_extended = DIFFERENCEMAP_EXTENDED - nonlocal_extended
            local_extended = delayed(subtract)( DIFFERENCEMAP_EXTENDED, nonlocal_extended)

            # INTERPOLATE LOCAL EFFECTS TO NO-LCC GRID BOXES    
            #local_interped_extended = interpolate_local( CHESSBOARD_EXTENDED, local_extended, LAND_EXTENDED, GLACIER_EXTENDED, coastboxes)
            local_interped_extended = delayed(interpolate_local)( CHESSBOARD_EXTENDED, local_extended, LAND_EXTENDED, GLACIER_EXTENDED, coastboxes)
            
            local_timeseries = delayed(get_original_grid)(local_interped_extended, LON, step, local_timeseries)    
            
        else:
            print("The interpolation method: <", INTERPOLATE_METHOD, "> could not be identified... :|")
            sys.exit( 2)
            
        
        #return local_interped, local, non_local


    total = delayed(write_netcdf)( total_timeseries, non_local_timeseries, local_timeseries, f, UNIT, LAT_DIFF, NSTEPS, NLAT, NLON, INTERPOLATE_METHOD)
    #total = delayed(write_netcdf)( total_timeseries, non_local_timeseries, f, UNIT, LAT_DIFF, NSTEPS, NLAT, NLON)
    total.compute()
else:##i.e. model==ecearth
    ##create global files here, in loop the signal seperation is performed for all land pixels only, after loop data is transferred from land grid to global grid
    non_local_timeseries=np.zeros([NSTEPS,len(CHESSBOARD)])
    total_timeseries=np.zeros([NSTEPS,len(CHESSBOARD)])
    local_timeseries=np.zeros([NSTEPS,len(CHESSBOARD)])
    
    print(NSTEPS)    
    for step in range(NSTEPS):
        differencemap = cp(DIFFERENCEMAP_ALLTIMESTEPS[step,:])
        
        ##put interpolation in function as for other esms so we can use delayed function, also write 2 new write funcs one for reduced gaussion grid and one for gaussian
        #plt.scatter(CHESSBOARD_LON,CHESSBOARD_LAT,c=CHESSBOARD,cmap='coolwarm',s=1,edgecolors='none',vmin=0,vmax=1)
        #plt.colorbar()
        #plt.show()
        
        #plt.scatter(CHESSBOARD_LON,CHESSBOARD_LAT,c=coastboxes_checkers,cmap='coolwarm',s=1,edgecolors='none',vmin=0,vmax=1)
        #plt.colorbar()
        #plt.show()
        
        ##only retain values over land as we only have the checkerboard defined over land (note that this means we do not need to add extra lon at edge)
        diff_map=differencemap[translation_table[:,0]]
        #---------coordinate of the grid boxes that are interpolated
        points = np.where( np.all( np.array(( CHESSBOARD == 0)), axis = 0))
        points_x=CHESSBOARD_LAT[CHESSBOARD == 0]
        points_y=CHESSBOARD_LON[CHESSBOARD == 0]
        values = diff_map[CHESSBOARD == 0]   ##issue here since dfmap is global but should first be reduced to checkers grid
        #values = differencemap_extended[points]
        
        #grid_x, grid_y = np.mgrid[0:96:1, 0:land_extended.shape[1]:1]
        grid_x = CHESSBOARD_LAT
        grid_y = CHESSBOARD_LON
        interped = interpolate.griddata( np.array(( points_y * 1.0, points_x * 1.0)).T, values,(grid_y,grid_x), method='linear')
        interped_nearest = interpolate.griddata( np.array(( points_y * 1.0, points_x * 1.0)).T, values,(grid_y,grid_x), method='nearest')
        non_local = cp( interped)
        non_local[coastboxes_checkers == 1] = interped_nearest[coastboxes_checkers == 1]
        non_local_timeseries[step,:]=non_local
        # take values over the ocean directly
        #non_local[land_extended == 0] = differencemap_extended[land_extended == 0]
        
        #plt.scatter(CHESSBOARD_LON[coastboxes_checkers==1],CHESSBOARD_LAT[coastboxes_checkers==1],c=non_local[coastboxes_checkers==1],cmap='coolwarm',s=1,edgecolors='none',vmin=-2,vmax=2)
        #plt.colorbar()
        #plt.show()
        
        #plt.scatter(CHESSBOARD_LON,CHESSBOARD_LAT,c=non_local_timeseries[step,:],cmap='coolwarm',s=1,edgecolors='none',vmin=-2,vmax=2)
        #plt.colorbar()
        #plt.show()
        
        #plt.scatter(CHESSBOARD_LON,CHESSBOARD_LAT,c=non_local-non_local_timeseries[step,:],cmap='coolwarm',s=1,edgecolors='none',vmin=-2,vmax=2)
        #plt.colorbar()
        #plt.show()
        
         #---------coordinate of the grid boxes that are interpolated
        points_x=CHESSBOARD_LAT[CHESSBOARD != 0]
        points_y=CHESSBOARD_LON[CHESSBOARD != 0]
        values = diff_map[CHESSBOARD != 0] 
        
        interped = interpolate.griddata( np.array(( points_y * 1.0, points_x * 1.0)).T, values,(grid_y,grid_x), method='linear')
        interped_nearest = interpolate.griddata( np.array(( points_y * 1.0, points_x * 1.0)).T, values,(grid_y,grid_x), method='nearest')
        
        total_timeseries_interpolated = cp( interped)
        # interpolate values at the coast differently
        total_timeseries_interpolated[coastboxes_checkers == 1] = interped_nearest[coastboxes_checkers == 1]
        total_timeseries[step,:]=total_timeseries_interpolated

        local_diff_map=total_timeseries_interpolated-non_local
        values=local_diff_map[CHESSBOARD != 0] 
        
        interped = interpolate.griddata( np.array(( points_y * 1.0, points_x * 1.0)).T, values,(grid_y,grid_x), method='linear')
        interped_nearest = interpolate.griddata( np.array(( points_y * 1.0, points_x * 1.0)).T, values,(grid_y,grid_x), method='nearest')
        
        local_timeseries_interpolated = cp( interped)
        # interpolate values at the coast differently
        local_timeseries_interpolated[coastboxes_checkers == 1] = interped_nearest[coastboxes_checkers == 1]
        local_timeseries[step,:]=local_timeseries_interpolated
        
        
        
        #plt.scatter(CHESSBOARD_LON,CHESSBOARD_LAT,c=diff_map,cmap='viridis',s=1,edgecolors='none')
        #plt.colorbar()
        #plt.show()
        
        #plt.scatter(CHESSBOARD_LON,CHESSBOARD_LAT,c=local_timeseries[step,:],cmap='viridis',s=1,edgecolors='none')
        #plt.colorbar()
        #plt.show()
        
        #plt.scatter(CHESSBOARD_LON,CHESSBOARD_LAT,c=non_local_timeseries[step,:]-total_timeseries[step,:],cmap='viridis',s=1,edgecolors='none')
        #plt.colorbar()
        #plt.show()
        
        #plt.scatter(CHESSBOARD_LON,CHESSBOARD_LAT,c=non_local_timeseries[step,:],cmap='coolwarm',s=1,edgecolors='none',vmin=-2,vmax=2)
        #plt.colorbar()
        #plt.show()
        
        #plt.scatter(CHESSBOARD_LON,CHESSBOARD_LAT,c=total_timeseries[step,:],cmap='viridis',s=1,edgecolors='none')
        #plt.colorbar()
        #plt.show()
    
        #add non local signal in raw output and overwrite only on land pixels while keeping original values over ocean for non-local and total signals
    full_local_timeseries=np.zeros([NSTEPS,len(LAT)])
    full_total_timeseries=cp(DIFFERENCEMAP_ALLTIMESTEPS)
    full_non_local_timeseries=cp(DIFFERENCEMAP_ALLTIMESTEPS)
    #test=DIFFERENCEMAP_ALLTIMESTEPS
    for t in range(len(CHESSBOARD_LAT)):
            full_non_local_timeseries[:,translation_table[t,0]]=non_local_timeseries[:,t]
            full_local_timeseries[:,translation_table[t,0]]=local_timeseries[:,t]
            full_total_timeseries[:,translation_table[t,0]]=total_timeseries[:,t]
    #test[:,translation_table[:,0]]=np.zeros([NSTEPS,len(CHESSBOARD_LAT)])
    #full_non_local_timeseries[:,translation_table[:,0]]=non_local_timeseries[:,:]
    #full_total_timeseries[:,translation_table[:,0]]=total_timeseries[:,:]
    #full_local_timeseries[:,translation_table[:,0]]=local_timeseries[:,:]
    
    #full_non_local_timeseries=np.zeros([NSTEPS,len(LAT)])
    #for i in range(len(LAT)):
    #    if i in translation_table[:,0]:
    #        full_non_local_timeseries[:,i]=non_local_timeseries[:,np.where(translation_table[:,0] == i)[0]][0]
            #full_total_timeseries[:,translation_table[j,0]]=total_timeseries[:,j]
            #full_local_timeseries[:,translation_table[j,0]]=local_timeseries[:,j]
    #    else:
    #        full_non_local_timeseries[:,i]=diff_map[:,i]
    
    #rint(non_local_timeseries[0,:])
    #print(full_non_local_timeseries[0,translation_table[:,0]])
    
    
        
    #plt.scatter(LON,LAT,c=full_local_timeseries[0,:],cmap='coolwarm',s=1,edgecolors='none',vmin=-2,vmax=2)
    #plt.colorbar()
    #plt.show()
    
    #plt.scatter(LON,LAT,c=full_non_local_timeseries[0,:],cmap='coolwarm',s=1,edgecolors='none',vmin=-2,vmax=2)
    #plt.colorbar()
    #plt.show()
    
    
    #plt.scatter(LON,LAT,c=full_total_timeseries[0,:],cmap='coolwarm',s=1,edgecolors='none',vmin=-2,vmax=2)
    #plt.colorbar()
    #plt.show()
    
    
    #plt.scatter(LON,LAT,c=full_non_local_timeseries[0,:]-full_total_timeseries[0,:],cmap='coolwarm',s=1,edgecolors='none',vmin=-2,vmax=2)
    #plt.colorbar()
    #plt.show()

    print(full_total_timeseries.shape)
    ##here the code for thye write netcdf is given but adapted for the ecearth data reduced gaussian grid
    print("Writing netCDF4 file...")
    #ifile = FILE_DIFFERENCEMAP.replace(".nc","") + "_signal-separated.nc"
    #f = nc.Dataset( ifile, 'r+') # format='NETCDF4')
    # Define global attributes
    f.creation_date=str(dt.datetime.today())
    f.contact='felix.havermann@lmu.de; Ludwig-Maximilians-University Munich'
    f.comment='Produced with Script ' + os.path.basename(__file__)
    #f.title='Land cover fractions of MPIESM-1.2-LR LAMACLIMA ' + SIMULATION_TYPE + ' simulation'
    #f.subtitle=''
    #f.anything=''
    #f_restart.comment='Only the variable cover_fract_pot was changed to a 100 % forest world and all other cover types are set to zero.\n'\
    #f.comment='Only the variable cover_fract_pot was changed to a 100 % forest world and all other cover types are set to fract_small (1e-10).\n'\
        #'This file is used for a 100 % forest simulation within the LAMACLIMA project\n'\
        #'Produced with Script ' + os.path.basename(__file__)
    #f.createVariable(LCT_CONVERSION[key][1], np.float32,('lat','lon'), fill_value=FILL_VALUE)
    
        # NON-LOCAL SIGNAL 
    f.createVariable( VAR + "_nonlocal", np.float64,('time','rgrid'), fill_value=FILL_VALUE)
    #f.variables[VAR + "_nonlocal"][0,:,:] = non_local_timeseries[:,i::j,:]
    #f.variables[VAR + "_nonlocal"][1:,:,:] = np.zeros(( NSTEPS-1, NLAT, NLON))
    f.variables[VAR + "_nonlocal"][:,:] = full_non_local_timeseries[:,:]
    f.variables[VAR + "_nonlocal"].longname = 'Non-local (=remote) effect of LCLM change'
    f.variables[VAR + "_nonlocal"].units = UNIT
    f.variables[VAR + "_nonlocal"].grid_type = 'gaussian'
    

    if INTERPOLATE_METHOD == "interpolate_local":
        # TOTAL SIGNAL
        full_local_timeseries[np.isnan(full_local_timeseries)] = 0
        total_timeseries_summed_up = full_local_timeseries[:,:] + full_non_local_timeseries[:,:]
        f.createVariable( VAR + "_total", np.float64,('time','rgrid'), fill_value=FILL_VALUE)
        #f.variables[VAR + "_total"][0,:,:] = local_timeseries[:,:] + non_local_timeseries[:,:]
        f.variables[VAR + "_total"][:,:] = total_timeseries_summed_up
        #f.variables[VAR + "_total"][1:,:,:] = np.zeros(( NSTEPS-1, NLAT, NLON))
        f.variables[VAR + "_total"].longname = 'Total (=local + non-local) effect of LCLM change'
        f.variables[VAR + "_total"].units = UNIT
        f.variables[VAR + "_total"].grid_type = 'gaussian'
        
        # LOCAL SIGNAL
        #MASK = np.tile( CHESSBOARD[i::j,:], (NSTEPS,1,1))
        f.createVariable( VAR + "_local", np.float64,('time','rgrid'), fill_value=FILL_VALUE)
        #f.variables[VAR + "_local"][0,:,:] = local_interped[:,i::j,:]
        #f.variables[VAR + "_local"][0,:,:] = local_timeseries[i::j,:]
        #f.variables[VAR + "_local"][1:,:,:] = np.zeros(( NSTEPS-1, NLAT, NLON))
        f.variables[VAR + "_local"][:,:] =  full_local_timeseries[:,:]
        f.variables[VAR + "_local"].longname = 'Local effect of LCLM change'
        f.variables[VAR + "_local"].units = UNIT
        f.variables[VAR + "_local"].grid_type = 'gaussian'    
    
    
    elif INTERPOLATE_METHOD == "interpolate_total":
        # TOTAL SIGNAL
        f.createVariable( VAR + "_total", np.float64,('time','rgrid'), fill_value=FILL_VALUE)
        f.variables[VAR + "_total"][:,:] = full_total_timeseries[:,:]
        f.variables[VAR + "_total"].longname = 'Total effect of LCLM change'
        f.variables[VAR + "_total"].units = UNIT
        f.variables[VAR + "_total"].grid_type = 'gaussian'
    
        # LOCAL SIGNAL
        full_local_timeseries_difference = full_total_timeseries[:,:] - full_non_local_timeseries[:,:]
        
        #MASK = np.tile( CHESSBOARD[:], (NSTEPS,1,1))
        #MASK = np.tile( CHESSBOARD, (NSTEPS,1,1))
        f.createVariable( VAR + "_local", np.float64,('time','rgrid'), fill_value=FILL_VALUE)
        f.variables[VAR + "_local"][:,:] = full_local_timeseries_difference[:,:]
        f.variables[VAR + "_local"].longname = 'Local effect of LCLM change'
        f.variables[VAR + "_local"].units = UNIT
        f.variables[VAR + "_local"].grid_type = 'gaussian'
        
        # LOCAL SIGNAL        
        #MASK = np.tile( CHESSBOARD[:], (NSTEPS,1,1))
        #MASK = np.tile( CHESSBOARD, (NSTEPS,1,1))
        f.createVariable( VAR + "_real_local", np.float64,('time','rgrid'), fill_value=FILL_VALUE)
        f.variables[VAR + "_real_local"][:,:] = full_local_timeseries[:,:]
        f.variables[VAR + "_real_local"].longname = 'Local effect of LCLM change'
        f.variables[VAR + "_real_local"].units = UNIT
        f.variables[VAR + "_real_local"].grid_type = 'gaussian'

    #plt.scatter(LON,LAT,c=full_non_local_timeseries[4,:],cmap='coolwarm',s=1,edgecolors='none',vmin=-2,vmax=2)
    #plt.colorbar()
    #plt.show()
    
    #plt.scatter(LON,LAT,c=full_total_timeseries[4,:],cmap='coolwarm',s=1,edgecolors='none',vmin=-2,vmax=2)
    #plt.colorbar()
    #plt.show()
    
    #plt.scatter(LON,LAT,c=full_local_timeseries[4,:],cmap='coolwarm',s=1,edgecolors='none',vmin=-2,vmax=2)
    #plt.colorbar()
    #plt.show()
    
    #f.close()
    print("writing finished.")
    #create new zeros array with globval coverage, use translation table and loop and fill values up, where land sea mask ==0 just fill non local and total up with diffmap vals
    
        #final results achieved still to merge and write to netcdf
    #if False:
    
        #fig = plt.figure()  
        #plt.imshow(non_local, cmap="Wistia", interpolation= 'None')
        #plt.colorbar()
        #plt.show()
        
        #fig = plt.figure()  
        #plt.imshow(local, cmap="Wistia", interpolation= 'None')
        #plt.colorbar()
        #plt.show()

        #fig = plt.figure()  
        #plt.imshow(local_interped, cmap="Wistia", interpolation= 'None')
        #plt.colorbar()
        #plt.show()
        
        #fig = plt.figure()  
        #plt.imshow(local_interped_masked, cmap="Wistia", interpolation= 'None')
        #plt.colorbar()
        #plt.show()
        
        #fig = plt.figure()  
        #plt.imshow(SEA_LAND_MASK, cmap="Wistia",interpolation= 'None')
        ##cb = plt.colorbar(im, orientation='horizontal')
        #plt.colorbar()
        #plt.show()    
        
        #fig = plt.figure()  
        #plt.imshow(LAND_EXTENDED, cmap="Wistia",interpolation= 'None')
        ##cb = plt.colorbar(im, orientation='horizontal')
        #plt.colorbar()
        #plt.show()
        
        #fig = plt.figure()  
        #plt.imshow(CHESSBOARD_EXTENDED, cmap="Wistia",interpolation= 'None')
        ##cb = plt.colorbar(im, orientation='horizontal')
        #plt.colorbar()
        #plt.show()
        
        #fig = plt.figure()  
        #plt.imshow(GLOBAL_CHESSBOARD_EXTENDED, cmap="Wistia", interpolation= 'None')
        #plt.colorbar()
        #plt.show()
        
        #fig = plt.figure()  
        #plt.imshow(GLOBAL_CHESSBOARD_EXTENDED_REVERSE, cmap="Wistia", interpolation= 'None')
        #plt.colorbar()
        #plt.show()
        
        #fig = plt.figure()  
        #plt.imshow(coastboxes, cmap="Wistia", interpolation= 'None')
        #plt.colorbar()
        #plt.show()
        
        #fig = plt.figure()  
        #plt.imshow(coastboxes_2, cmap="Wistia", interpolation= 'None')
        #plt.colorbar()
        #plt.show()



    #if False:
        #fig = plt.figure()
        #plt.subplot(2,2,1)
        #plt.imshow(local_interped, interpolation= 'None')#,vmin=-2,vmax=2)
        #plt.colorbar()
        #plt.subplot(2,2,2)
        #plt.imshow(local_extended, interpolation= 'None')#,vmin=-2,vmax=2)
        #plt.colorbar()
        #plt.subplot(2,2,3)
        #plt.imshow(local_interped_extended, interpolation= 'None')#,vmin=-2,vmax=2)
        #plt.colorbar()
        #plt.subplot(2,2,4)
        #plt.imshow(CHESSBOARD_EXTENDED, interpolation= 'None')#,vmin=-2,vmax=2)
        #plt.colorbar()
        #plt.show()    

    #if False:
        #fig = plt.figure()
        #plt.subplot(2,2,1)
        #plt.imshow(local_interped, interpolation= 'None',vmin=-2,vmax=2)
        #plt.colorbar()
        #plt.subplot(2,2,2)
        #plt.imshow(local, interpolation= 'None',vmin=-2,vmax=2)
        #plt.colorbar()
        #plt.subplot(2,2,3)
        #plt.imshow(non_local, interpolation= 'None',vmin=-2,vmax=2)
        #plt.colorbar()
        #plt.subplot(2,2,4)
        #plt.imshow(interped_interp1, interpolation= 'None',vmin=-2,vmax=2)
        #plt.colorbar()
        #plt.show()
        
    
    #m = Basemap(projection='cyl', lon_0 = 0, resolution='c')
    ##meshlon, meshlat = m(LON, LAT)    
    
    #fig = plt.figure()  
    ##m.drawcoastlines(linewidth=0.5)
    ##m.drawparallels(np.arange(-20,61,20),labels=[0,1,0,0],fontsize=20, color='grey')
    
    ##m.drawparallels(np.arange(-80,81,40))#, labels=[0,1,0,0])
    ##m.drawmeridians(np.arange(-160,160,20), labels=[0,0,0,1])#,fontsize=8,linewidth=0,labelstyle="+/-")
    
    ##m.pcolormesh( LON, LAT, local_interped + non_local, latlon=True, cmap=cmap_bwr_jwi,\
        ##norm=norm,vmin=-plotmax, vmax=plotmax, rasterized=True, zorder=1)    
    
    ##im = m.pcolormesh( LON, LAT, local_interped, latlon=True, cmap=cmap_bwr_jwi,\
        ##norm=norm,vmin=-plotmax, vmax=plotmax, rasterized=True, zorder=1)    
    
    #im = m.pcolormesh(LON, LAT, CHESSBOARD, cmap='Wistia', latlon=True)# rasterized=True, zorder=1
    #im = m.pcolormesh(LON, LAT, differencemap, cmap='Wistia', latlon=True)# rasterized=True, zorder=1
    #im = m.pcolormesh(LON, LAT, non_local, cmap='Wistia', latlon=True)#, zorder=1)#, rasterized=True,
    #im = m.pcolormesh(LON, LAT, local, cmap='Wistia', latlon=True)#, zorder=1)#, rasterized=True, 
    #im = m.pcolormesh(LON, LAT, local_interped, cmap='Wistia', latlon=True)#, rasterized=True, zorder=1)
    
    #plt.show()
    #im = m.pcolormesh(ii - 0.05, jj - 0.05, interpolated_grid, cmap='bwr', vmin= -5, vmax= 5)
    #cb = plt.colorbar(im, orientation='horizontal')
    #plt.show()
