import numpy as np
import os, sys
import xarray as xr
import datetime as dt
from datetime import timedelta
import time
import cartopy.crs as ccrs

import lis_utils as utils
#from helpers import get_bounding_latlon_slice

use_jon_method = False
plotfd = False
plotraster = True
dogrid = True
dosubset = False
geo_subset = [-92.0, -79.0, 30.0, 40.0] # SEUS 2016 flash drought
smvi_thresh = .3 ## ratio of SMVI pixels per county for it to be "active"

DATADIR = f'/raid2/sport/people/casejl/LIS7/OUTPUT/conus3km/SURFACEMODEL'
#COUNTYDIR = f'{DATADIR}/county_percentile'
GRIDDIR = f'{DATADIR}/grid_percentile'
OUTDIR = '/usr/people/mdodson/sport-lis-smvi/figures/jon'

map_proj = ccrs.PlateCarree()
data_trans = ccrs.PlateCarree()

#region_list = ['nc', 'neus', 'matl', 'al', 'fl', 'tn', 'tx', 'in', 'midwest',
#               'dakotas', 'swus', 'nwus', 'ca', 'conus']

region_list = ["al", "ca", "dakotas", "fl", "in", "la", "matl", "midwest",
    "mt", "nc", "neus", "nwus", "swus", "conus"]

#region_list = ['conus', 'al', 'nc']
#region_list = ['mt']
#region_list = ['la', 'al', 'sedews']
#region_list = ['sedews']
#region_list = ['neus']

###################################
# Beginning of main
###################################

if __name__ == '__main__':
    # start and end dates
    edate = '20161120'

    if len(sys.argv) > 1:
        edate = sys.argv[1]

    eyyyy = edate[0:4]
    emo = edate[4:6]
    edd = edate[6:8]
    vdate = dt.datetime(int(eyyyy), int(emo), int(edd)).strftime('%d %b %Y')
    sdate = (dt.datetime(int(eyyyy), int(emo), int(edd))
             - timedelta(days=40)).strftime('%Y%m%d')
    ## ndays should always be 40 since 40 days explicitly encoded above
    ndays = utils.get_ndays(sdate, edate)
    print(f'sdate {sdate} edate {edate} vdate {vdate} ndays {ndays}')
    #sys.exit()

    if dogrid:
        dsis = [None] * ndays
        dsip = [None] * ndays
        dates = [None] * ndays
        yyyymmdd = sdate
        print(f'Reading LIS files from {sdate} to {edate}')
        for nn in range(1, ndays + 1):
#            print(f'Reading LIS_HIST for {yyyymmdd}')
            yyyy = yyyymmdd[0:4]
            mo = yyyymmdd[4:6]
            dd = yyyymmdd[6:8]
            soilfile = f'{DATADIR}/{yyyy}{mo}/LIS_HIST_{yyyymmdd}0000.d01.grb'
            #print(f'Opening {soilfile} into xarray')
            dsis[nn-1] = xr.open_mfdataset(f'{soilfile}', engine='pynio')
            dates[nn-1] = dt.datetime(int(yyyy), int(mo), int(dd), 00)
            datep1 = dates[nn-1] + timedelta(days=1)
            yyyymmdd = datep1.strftime('%Y%m%d')

        print(f'Concatenating files. This make take a while....')
        tstart = time.time()
        ds_soil = xr.concat(dsis, 'time', data_vars='all',
                            coords='minimal', compat='equals')
        #print(f'\n{ds_soil}\n')
#        print(f'\n{ds_soil.data_vars}\n')

        percfile = f'{GRIDDIR}/{eyyyy}/vsm_percentile_{edate}.grb2'
#        print(f'Opening {percfile} into xarray')
        ds_perc = xr.open_dataset(f'{percfile}', engine='pynio')
#        print(f'{ds_perc.data_vars}\n')
        tstop = time.time() - tstart
        #print(f' -time to concatenate files: {(tstop/60.0):0.1f} min')

        # Read in VSM and SMPERC arrays
        soil_grid = ds_soil.SOIL_M_GDS0_DBLY
        perc_grid = ds_perc.SMPERC_P0_2L106_GLL0

        # Dimenions
        print('\nSOIL Dimenions: ', soil_grid.dims)
        print('PERC Dimenions: ', perc_grid.dims)

        # Lats/Lons
        lats = perc_grid['lat_0'].values
        lons = perc_grid['lon_0'].values
        #print(f'\nmin max latitude: {lats.min()}, {lats.max()}')
        #print(f'min max longitude: {lons.min()}, {lons.max()}')

        # Metadata
        metadatas = soil_grid.attrs
        # print('\nBEFORE metadata: ', metadatas)
        soil_grid.attrs['center'] = 'NASA SPoRT Center'
        soil_grid.attrs['long_name'] = 'SPoRT-LIS analyses with Noah LSM'
        soil_grid.attrs['model'] = 'SPoRT-LIS'
        soil_grid.attrs['units'] = 'm3/m3'
        # print('\nAFTER metadata: ', metadatas)
        #print('\n SOIL_GRID: \n', soil_grid)
        metadatap = perc_grid.attrs
        # print('\nBEFORE metadata: ', metadatap)
        perc_grid.attrs['center'] = 'NASA SPoRT Center'
        perc_grid.attrs['long_name'] = 'Gridded Soil Moisture Percentile'
        perc_grid.attrs['model'] = 'SPoRT-LIS'
        # print('\nAFTER metadata: ', metadatap)
        #print('\n PERC_GRID: \n', perc_grid)

        slevels = soil_grid['lv_DBLY2'].values
        print(f'\nLevels in the SOIL data: {slevels}')
        times = soil_grid['time'].values
        ntimes = times.max() + 1
        print(f'ntimes: {ntimes}; SOIL min max times: {times.min()}, {times.max()} {ntimes}')
        plevels = perc_grid['lv_DBLL0'].values
        print(f'Levels in the PERC data: {plevels}')

        # SPoRT-LIS geo extent
        lllon = lons.min()
        urlon = lons.max()
        lllat = lats.min()
        urlat = lats.max()
        geoExtent = [lllon, urlon, lllat, urlat]
        print(f'Global geoExtent from percentiles grb2: {geoExtent}')
        if dosubset:
            geoExtent = geo_subset
            lllon = geoExtent[0]
            urlon = geoExtent[1]
            lllat = geoExtent[2]
            urlat = geoExtent[3]

        print(f'geoExtent: {geoExtent}')

        print(f'\nStoring as numpy arrays. May take a while....')
        tstart = time.time()

        vsm1 = soil_grid.isel(lv_DBLY2=0)
        vsm2 = soil_grid.isel(lv_DBLY2=1)
        vsm3 = soil_grid.isel(lv_DBLY2=2)
        vsm4 = soil_grid.isel(lv_DBLY2=3)
        #print(f'{vsm1} \n{vsm2} \n{vsm3} \n{vsm4}')
        vsm11 = vsm1
        vsm12 = (10.0*vsm1 + 30.0*vsm2) / 40.0
        vsm13 = (10.0*vsm1 + 30.0*vsm2 + 60.0*vsm3) / 100.0
        vsm14 = (10.0*vsm1 + 30.0*vsm2 + 60.0*vsm3 + 100.0*vsm4) / 200.0

        perc11 = perc_grid.isel(lv_DBLL0=0).to_numpy()
        perc12 = perc_grid.isel(lv_DBLL0=1).to_numpy()
        perc13 = perc_grid.isel(lv_DBLL0=2).to_numpy()
        perc14 = perc_grid.isel(lv_DBLL0=3).to_numpy()
        nlats = perc11.shape[0]
        nlons = perc11.shape[1]
#        print(f'percentile dims: {nlats} {nlons}')
        #print(vsm11, vsm12, vsm13, vsm14)
        #print(perc11, perc12, perc13, perc14)

        tstop = time.time() - tstart
#        print(f' [time to convert to numpy arrays: {tstop:0.1f} sec]')

        # Compute running means
        ## only runs once w/ ( nn = ntimes-1 = soil_grid['time'].value.max() )
        ## based on hard-coded value above, nn always equals 40
        for nn in range(ntimes-1, ntimes):
            ## each vsm array shaped (T,Y,X) = (41, 929, 1929)
            print("NN:",nn)
            fd11, fd12, fd13, fd14 = \
                utils.get_fd_array(vsm11, vsm12, vsm13, vsm14, nlats, nlons, nn,
                        jon_method=use_jon_method)
        print(fd11.shape, fd12.shape, fd13.shape, fd14.shape)
        exit(0)

        # Check that percentiles fell below 20. If so, set flash_drought arrays
        # to 2.0 to signify flash drought criteria fully met.
        # May want to pythonize this section, but the hang-up is mostly with
        # xr.open_msfdataset() and setting arrays to .values or .to_numpy.
        start = time.time()
        fd11[(perc11 < 20.0) & (fd11 > 0)] = 2.0
        fd12[(perc12 < 20.0) & (fd12 > 0)] = 2.0
        fd13[(perc13 < 20.0) & (fd13 > 0)] = 2.0
        fd14[(perc14 < 20.0) & (fd14 > 0)] = 2.0

        ctime = time.time() - start
        print(f' [time to finalize flash drought indicator: {ctime:0.2f} sec]')

    if plotfd:
        title1 = (f'SPoRT-LIS 0-10 cm Soil Moisture Percentile valid {vdate} '
                  f'\n (hatching: Soil Moisture Volatility Index flash drought criteria met)')
        title2 = (f'SPoRT-LIS 0-40 cm Soil Moisture Percentile valid {vdate} '
                  f'\n (hatching: Soil Moisture Volatility Index flash drought criteria met)')
        title3 = (f'SPoRT-LIS 0-100 cm Soil Moisture Percentile valid {vdate} '
                  f'\n (hatching: Soil Moisture Volatility Index flash drought criteria met)')
        title4 = (f'SPoRT-LIS 0-200 cm Soil Moisture Percentile valid {vdate} '
                  f'\n (hatching: Soil Moisture Volatility Index flash drought criteria met)')
        for region in region_list:
            print(f'\nPlotting region {region}')
            geo = utils.geog[region]
            print(f'  -{edate} 0-10 cm VSM Percentiles with FD flags')
            fname1 = f'{OUTDIR}/vsm0-10percent_{edate}_00z_{region}.png'
            utils.make_fd_plots(lons, lats, perc11, fd11, geo, title1, fname1, region)
            print(f'  -{edate} 0-40 cm VSM Percentiles with FD flags')
            fname2 = f'{OUTDIR}/vsm0-40percent_{edate}_00z_{region}.png'
            utils.make_fd_plots(lons, lats, perc12, fd12, geo, title2, fname2, region)
            print(f'  -{edate} 0-100 cm VSM Percentiles with FD flags')
            fname3 = f'{OUTDIR}/vsm0-100percent_{edate}_00z_{region}.png'
            utils.make_fd_plots(lons, lats, perc13, fd13, geo, title3, fname3, region)
            print(f'  -{edate} 0-200 cm VSM Percentiles with FD flags')
            fname4 = f'{OUTDIR}/vsm0-200percent_{edate}_00z_{region}.png'
            utils.make_fd_plots(lons, lats, perc14, fd14, geo, title4, fname4, region)

    if plotraster:
        title1 = (f'0-10 cm SMVI Flash Drought County Activation {vdate} ')
        title2 = (f'0-40 cm SMVI Flash Drought County Activation {vdate} ')
        title3 = (f'0-100 cm SMVI Flash Drought County Activation {vdate} ')
        title4 = (f'0-200 cm SMVI Flash Drought County Activation {vdate} ')
        for region in region_list:
            fname1 = f'{OUTDIR}/county0-10-fd_{edate}_00z_{region}.png'
            fname2 = f'{OUTDIR}/county0-40-fd_{edate}_00z_{region}.png'
            fname3 = f'{OUTDIR}/county0-100-fd_{edate}_00z_{region}.png'
            fname4 = f'{OUTDIR}/county0-200-fd_{edate}_00z_{region}.png'
            print(f'\nPlotting region {region}')
            geo = utils.geog[region]
            print(f'  -{edate} 0-10 cm County Raster FD Activation')
            utils.make_countyfd_plots(
                    lons, lats, fd11, geo, title1, fname1, region, smvi_thresh)
            print(f'  -{edate} 0-40 cm County Raster FD Activation')
            utils.make_countyfd_plots(
                    lons, lats, fd12, geo, title2, fname2, region, smvi_thresh)
            print(f'  -{edate} 0-100 cm County Raster FD Activation')
            utils.make_countyfd_plots(
                    lons, lats, fd13, geo, title3, fname3, region, smvi_thresh)
            print(f'  -{edate} 0-200 cm County Raster FD Activation')
            utils.make_countyfd_plots(
                    lons, lats, fd14, geo, title4, fname4, region, smvi_thresh)


sys.exit()
