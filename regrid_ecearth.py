import xarray as xr 
import numpy as np
import pickle
import scipy.interpolate as interpolate
import netCDF4 as nc
import matplotlib.pyplot as plt

LAT=pickle.load(open('/pf/b/b381334/lat_ec_earth.sav', 'rb'))
LON=pickle.load(open('/pf/b/b381334/lon_ec_earth.sav', 'rb'))
outdir='/scratch/b/b381334/signal_separation/'

cmor='Amon'
var='hurs'

#ls= ['ensmean','timmean000000','timmean000001','timmean000002','timmean000003','timmean000004']
ls=['DJFensmean','JJAensmean','MAMensmean','ensmean','SONensmean','timmean000000','timmean000001','timmean000002','timmean000003','timmean000004','ymonmean000000','ymonmean000001','ymonmean000002','ymonmean000003','ymonmean000004']
#ls=['ymonensmean']
#for case in ['crop-ctl','crop-frst','irr-crop','frst-ctl']:
for case in ['frst-ctl','crop-ctl','irr-crop']:
    for part in ls:
        print('writing interpolated file for '+case)
        #interpolate to regular grid 
        grid_x, grid_y = np.mgrid[-90:90:180/192,-180:180:1.25]
        print(outdir+case+'/'+cmor+'/'+var+'/'+var+'_'+case+'_ecearth_'+part+'_signal-separated.nc')
        ds=xr.open_dataset(outdir+case+'/'+cmor+'/'+var+'/'+var+'_'+case+'_ecearth_'+part+'_signal-separated.nc')
        local=ds[var+'_local']
        non_local=ds[var+'_nonlocal']
        total=ds[var+'_total']
        points_x=LAT[:,0]
        points_y=LON[:,0]
        if part in ['ymonmean000000','ymonmean000001','ymonmean000002','ymonmean000003','ymonmean000004','ymonensmean']:
            local_interped=np.zeros((12,len(grid_x[:,0]),len(grid_x[0,:])))
            non_local_interped=np.zeros((12,len(grid_x[:,0]),len(grid_x[0,:])))
            total_interped=np.zeros((12,len(grid_x[:,0]),len(grid_x[0,:])))
            for l in range(12): 
                local_interped[l,:,:]= interpolate.griddata( np.array(( points_y * 1.0, points_x * 1.0)).T, local[l,:],(grid_y,grid_x), method='linear')
                non_local_interped[l,:,:]= interpolate.griddata( np.array(( points_y * 1.0, points_x * 1.0)).T, non_local[l,:],(grid_y,grid_x), method='linear')
                total_interped[l,:,:]= interpolate.griddata( np.array(( points_y * 1.0, points_x * 1.0)).T, total[l,:],(grid_y,grid_x), method='linear')
        else:
            local_interped=np.zeros((len(grid_x[:,0]),len(grid_x[0,:])))
            non_local_interped=np.zeros((len(grid_x[:,0]),len(grid_x[0,:])))
            total_interped=np.zeros((len(grid_x[:,0]),len(grid_x[0,:])))
            local_interped= interpolate.griddata( np.array(( points_y * 1.0, points_x * 1.0)).T, local[0,:],(grid_y,grid_x), method='linear')
            non_local_interped= interpolate.griddata( np.array(( points_y * 1.0, points_x * 1.0)).T, non_local[0,:],(grid_y,grid_x), method='linear')
            total_interped= interpolate.griddata( np.array(( points_y * 1.0, points_x * 1.0)).T, total[0,:],(grid_y,grid_x), method='linear')


        fn = outdir+case+'/'+cmor+'/'+var+'/interped_'+var+'_'+case+'_ecearth_'+part+'_signal-separated.nc'
        ds = nc.Dataset(fn, 'w', format='NETCDF4')
       
        lat = ds.createDimension('lat', None)
        lon = ds.createDimension('lon', None)
        #times = ds.createVariable('time', 'f4', ('time',))
        lats = ds.createVariable('lat', 'f4', ('lat',))
        lons = ds.createVariable('lon', 'f4', ('lon',))
        lats[:] = np.arange(-90, 90, 180/192)
        lons[:] = np.arange(-180, 180, 1.25)
        if part in ['ymonmean000000','ymonmean000001','ymonmean000002','ymonmean000003','ymonmean000004','ymonensmean']:
            time = ds.createDimension('time', None)
            time = ds.createVariable('time', 'f4', ('time',))
            time[:]=np.arange(1,12.5,1)
            value = ds.createVariable(var + "_local", 'f4', ('time','lat', 'lon',))
            value[:, :,:] = local_interped[:,:,:]
            value = ds.createVariable(var + "_nonlocal", 'f4', ('time','lat', 'lon',))
            value[ :, :,:] = non_local_interped[:,:,:]
            value = ds.createVariable(var + "_total", 'f4', ('time','lat', 'lon',))
            value[ :, :,:] = total_interped[:,:,:]
        else:        
            #value = ds.createVariable('2t' + "_local", 'f4', ('time', 'lat', 'lon',))
            value = ds.createVariable(var + "_local", 'f4', ('lat', 'lon',))
            value[:, :] = local_interped[:,:]
            value = ds.createVariable(var + "_nonlocal", 'f4', ('lat', 'lon',))
            value[ :, :] = non_local_interped[:,:]
            value = ds.createVariable(var + "_total", 'f4', ('lat', 'lon',))
            value[ :, :] = total_interped[:,:]
        ds.close()

