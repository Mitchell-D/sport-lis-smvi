import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from datetime import timedelta
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
import sys
import pickle
import matplotlib.colors as col
import matplotlib.cm as cm
#import geopandas as gpd
import time
import xarray as xr

## mitchell's imports...
import geopandas as gpd
## probably a way to combine this w/ your formatters
from cartopy.mpl.ticker import LatitudeFormatter,LongitudeFormatter
import shapely ## needed to check version
from shapely.geometry import Point,Polygon
from shapely.strtree import STRtree

# Read in shapefile only once upfront.
county_shp = f'/raid2/sport/people/casejl/DATA/SHAPEFILES/c_15au13.shp'
counties = shpreader.Reader(county_shp)
#counties = gpd.read_file(county_shp)

POLY_RASTER_DIR = "/usr/people/mdodson/sport-lis-smvi/data/poly"

map_proj = ccrs.PlateCarree()
data_trans = ccrs.PlateCarree()
mapscale ='50m'
coastlines = cfeature.NaturalEarthFeature(
    category='physical',
    name='coastline',
    scale=mapscale,
    edgecolor='black',
    facecolor='none')
countries = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_countries',
    scale=mapscale,
    edgecolor='black',
    facecolor='none')
states = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces',
    scale=mapscale,
    edgecolor='black',
    facecolor='none')

pickleColorFile = '/usr/people/casejl/scripts/python/colors/gradscolors_37.pkl'
fp = open(pickleColorFile, 'rb')
c = pickle.load(fp)
fp.close()

clevs = [
    0, 2, 5, 10, 20, 30, 70, 80, 90, 95, 98, 100
]
clevs_drought = [
    2, 5, 10, 20, 30
]

# RGBs
d4 = c['94']
d3 = c['93']
d2 = c['92']
d1 = c['91']
d0 = c['90']
dn = c['87']
dnlg = c['89']
w0 = c['41']
w1 = c['43']
w2 = c['45']
w3 = c['47']
w4 = c['49']

fillcols_pm = [
    d4, d4, d3, d3, d3, d2, d2, d2, d2, d2, d1, d1, d1, d1, d1,
    d1, d1, d1, d1, d1, d0, d0, d0, d0, d0, d0, d0, d0, d0, d0,
    dn, dn, dn, dn, dn, dn, dn, dn, dn, dn, dn, dn, dn, dn, dn,
    dn, dn, dn, dn, dn, dn, dn, dn, dn, dn, dn, dn, dn, dn, dn,
    dn, dn, dn, dn, dn, dn, dn, dn, dn, dn,
    w0, w0, w0, w0, w0, w0, w0, w0, w0, w0, w1, w1, w1, w1, w1,
    w1, w1, w1, w1, w1, w2, w2, w2, w2, w2, w3, w3, w3, w4, w4
]
#cmap2 = col.LinearSegmentedColormap.from_list('own2', fillcols_pm)
fills_drought = [d3, d2, d1, d0]
cmap2 = ListedColormap(fills_drought)

geog = {
    'nc': [-85.70, -74.50, 31.75, 37.50],
    'neus': [-86.60, -66.50, 37.16, 47.52],
    'matl': [-84.10, -70.90, 35.70, 41.40],
    'al': [-92.50, -81.00, 29.80, 35.50],
    'fl': [-88.90, -78.53, 24.08, 32.30],
    'tn': [-92.30, -80.30, 34.40, 39.30],
    'tx': [-100.0, -88.50, 27.00, 32.80],
    'la': [-100.0, -87.8, 27.5, 34.0],
    'in': [-92.30, -78.80, 37.00, 42.90],
    'midwest': [-105.70, -86.55, 35.93, 45.75],
    'dakotas': [-107.10, -93.00, 42.40, 49.40],
    'swus': [-116.00, -99.00, 30.00, 38.00],
    'nwus': [-125.00, -104.00, 40.00, 50.00],
    'ca': [-125.50, -110.50, 31.99, 43.20],
    'mt': [-118.00, -100.00, 42.50, 50.00],
    'conus': [-126.00, -67.00, 24.90, 51.00],
    'sedews': [-92, -72, 25.1, 39.5],
}

# cbarpos info: [xll, yll, xlen, ylen]
cbarpos = {
    'nc': [0.13, 0.15, 0.76, 0.02],
    'neus': [0.13, 0.15, 0.76, 0.02],
    'matl': [0.13, 0.15, 0.76, 0.02],
    'al': [0.13, 0.15, 0.76, 0.02],
    'fl': [0.13, 0.05, 0.76, 0.02],
    'tn': [0.13, 0.15, 0.76, 0.02],
    'tx': [0.13, 0.15, 0.76, 0.02],
    'la': [0.13, 0.11, 0.76, 0.02],
    'in': [0.13, 0.15, 0.76, 0.02],
    'midwest': [0.13, 0.15, 0.76, 0.02],
    'dakotas': [0.13, 0.15, 0.76, 0.02],
    'swus': [0.13, 0.15, 0.76, 0.02],
    'nwus': [0.13, 0.15, 0.76, 0.02],
    'ca': [0.13, 0.05, 0.76, 0.02],
    'mt': [0.13, 0.15, 0.76, 0.02],
    'conus': [0.13, 0.15, 0.76, 0.02],
    'sedews': [0.17, 0.04, 0.68, 0.02],
}


##########################################
# Functions / Methods
##########################################

def get_ndays(syyyymmdd, eyyyymmdd):
    eyyyy = eyyyymmdd[0:4]
    emo = eyyyymmdd[4:6]
    edd = eyyyymmdd[6:8]
    yyyymmdd = syyyymmdd
    eyyyymmddp1 = (dt.datetime(int(eyyyy), int(emo), int(edd)) +
                   timedelta(days=1)).strftime('%Y%m%d')
    ndays = 0
    while yyyymmdd != eyyyymmddp1:
        ndays += 1
        yyyy = yyyymmdd[0:4]
        mo = yyyymmdd[4:6]
        dd = yyyymmdd[6:8]
        currdate = dt.datetime(int(yyyy), int(mo), int(dd))
        datep1 = currdate + timedelta(days=1)
        yyyymmdd = datep1.strftime('%Y%m%d')
    return ndays


def make_plots(lon, lat, var, geo, title, png):
#    fig, ax = plt.subplots(1, 1, figsize=(12, 9),
    fig, ax = plt.subplots(1, 1, figsize=(10, 8),
                           subplot_kw={'projection': map_proj})
    ax.set_extent(geo, crs=map_proj)
    # Draw Lats, Lons, Coastlines, Countries, States, and Lakes
    # using Natural Earth (public domain http://naturalearthdata.com)
    ax.add_feature(coastlines, linewidth=0.3,)
    ax.add_feature(countries, linewidth=0.5,)
    ax.add_feature(states, linewidth=0.5,)
    cs = ax.pcolormesh(lon, lat, var, cmap=cmap2)
    cs.cmap.set_under(d4)
    cs.cmap.set_over(w4)
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.10, 0.15, 0.80, 0.02])
    fig.colorbar(cs, cax=cbar_ax, ticks=clevs,
                 orientation='horizontal', extendfrac=0.05)
    ax.set(title=title)
    plt.savefig(png)
    plt.close()


def make_fd_plots(lon, lat, var, fd, geo, title, png, zoom):
    start = time.time()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8),
                           subplot_kw={'projection': map_proj})
    ax.set_extent(geo, crs=map_proj)
    # Draw Lats, Lons, Coastlines, Countries, States, and Lakes
    # using Natural Earth (public domain http://naturalearthdata.com)
    ax.add_feature(coastlines, linewidth=0.5,)
    ax.add_feature(countries, linewidth=0.5,)
    ax.add_feature(states, linewidth=0.5,)
    # Subset data before plotting to speed up plot generation
    lon_min = geo[0]; lon_max = geo[1]
    lat_min = geo[2]; lat_max = geo[3]
    lon_norm = ((lon + 180) % 360) - 180
    lon_idx = (lon_norm >= lon_min) & (lon_norm <= lon_max)
    lat_idx = (lat >= lat_min) & (lat <= lat_max)
    lon_sub = lon[lon_idx]
    lat_sub = lat[lat_idx]
    var_sub = var[np.ix_(lat_idx, lon_idx)]
    cs = ax.contourf(lon_sub, lat_sub, var_sub, levels=clevs_drought, colors=fills_drought, extend='both')
    ctime = time.time() - start
#    print(f'  --Lapsed time (percentile contourf): {ctime:0.2f} sec')
    cs.cmap.set_under(d4)
#    cs.cmap.set_over(w4)
    cs.cmap.set_over(dnlg)
    edgelw = 0.0
    hatchlw = 0.9
    hatch = ['xxx']
    if zoom != 'conus':
        edgelw = 0.05
        hatchlw = 0.5
        hatch = ['xx']
        # J.Case (12/4/2025) -- thin county lines as in nldas_utils.py on Discover
        ax.add_geometries(counties.geometries(), crs=map_proj, facecolor='none',
                          edgecolor='gray', linewidth=0.2, linestyle='-')
        ctime = time.time() - start
#        print(f'  --Lapsed time (shapefile plot): {ctime:0.2f} sec')

    fd_sub = fd[np.ix_(lat_idx, lon_idx)]
    hatch = ax.contourf(lon_sub, lat_sub, fd_sub, levels=[1.5, 2.5], colors=['none'], hatches=hatch)
    for collection in hatch.collections:
        collection.set_edgecolor('black')
        collection.set_linewidth(edgelw)
        plt.rcParams['hatch.linewidth'] = hatchlw
    ctime = time.time() - start
#    print(f'  --Lapsed time (hatch plot): {ctime:0.2f} sec')

    lat_inc, lon_inc = _find_latlon_inc(geog[zoom])
    _draw_grid_lines(ax, map_proj, geog[zoom], lon_inc, lat_inc, lw=0.1, ls=":", label=True)

    # for cbar: [left, bottom, width, height]
    cbar_ax = fig.add_axes(cbarpos[zoom])
    fig.colorbar(cs, cax=cbar_ax, ticks=clevs,
                 orientation='horizontal', extendfrac=0.05)
    ax.set(title=title)
    ctime = time.time() - start
#    print(f'  --Lapsed time (misc plots): {ctime:0.2f} sec')

    fig.savefig(png, dpi=100)
    plt.close(fig)
    ctime = time.time() - start
#    print(f'  --Lapsed time (plot close): {ctime:0.2f} sec')


def get_fd_array(v11, v12, v13, v14, nlat, nlon, nt, jon_method=True):
    # NOTE: v11, v12, v13, and v14 should be xr DataArrays.
    counts1 = np.zeros((nlat, nlon), dtype=np.int32)
    counts2 = np.zeros((nlat, nlon), dtype=np.int32)
    counts3 = np.zeros((nlat, nlon), dtype=np.int32)
    counts4 = np.zeros((nlat, nlon), dtype=np.int32)
    fd1 = np.zeros((nlat, nlon))
    fd2 = np.zeros((nlat, nlon))
    fd3 = np.zeros((nlat, nlon))
    fd4 = np.zeros((nlat, nlon))

    start = time.time()

    # combined has shape: (var=4, time, y, x)
    combined = xr.concat([v11, v12, v13, v14], dim="var")
    # shape: (4, time, y, x)
    combined_np = combined.values
    v11_np = combined_np[0]
    v12_np = combined_np[1]
    v13_np = combined_np[2]
    v14_np = combined_np[3]
    v11_05d = _rolling_mean_np(v11_np, 5)
    v12_05d = _rolling_mean_np(v12_np, 5)
    v13_05d = _rolling_mean_np(v13_np, 5)
    v14_05d = _rolling_mean_np(v14_np, 5)
    v11_20d = _rolling_mean_np(v11_np, 20)
    v12_20d = _rolling_mean_np(v12_np, 20)
    v13_20d = _rolling_mean_np(v13_np, 20)
    v14_20d = _rolling_mean_np(v14_np, 20)

    ctime = time.time() - start
    print(f' [time to convert to numpy arrays: {ctime:0.1f} sec]')

    ## 5-day averages have 36 timesteps; 20-day has 21 timesteps
    ## the final time step of both arrays refers to the respective windows'
    ## averages up to and including the end day.
    print(nt, combined_np.shape, v11_05d.shape, v11_20d.shape)

    day_thresh = 20
    if jon_method:
        ## iterate from 20 to 39, inclusively. This ignores the last day, which
        ## may be unintended.
        for nn in range(nt - day_thresh, nt):
            print(f'  -Evaluating 5/20 day running means at time t={nn}.')
            # Rolling 5-day mean: index nn-window (numpy method)
            ## indexes 15-34 inclusive, which refer to 5-day moving averages
            ## ending with days in the range [start - 20, start - 1]
            v11_05d_avg = v11_05d[nn - 5]
            v12_05d_avg = v12_05d[nn - 5]
            v13_05d_avg = v13_05d[nn - 5]
            v14_05d_avg = v14_05d[nn - 5]
            # Rolling 20-day mean
            ## indexes 0-19 inclusive, which refer to 20-day moving averages
            ## ending with days in the range [start - 20, start - 1]
            v11_20d_avg = v11_20d[nn - 20]
            v12_20d_avg = v12_20d[nn - 20]
            v13_20d_avg = v13_20d[nn - 20]
            v14_20d_avg = v14_20d[nn - 20]
            counts1 += (v11_05d_avg < v11_20d_avg).astype(np.int32)
            counts2 += (v12_05d_avg < v12_20d_avg).astype(np.int32)
            counts3 += (v13_05d_avg < v13_20d_avg).astype(np.int32)
            counts4 += (v14_05d_avg < v14_20d_avg).astype(np.int32)

        print(f'  -Finding points that meet flash drought criteria....')
        fd1[counts1 > day_thresh - 1] = 1.0
        fd2[counts2 > day_thresh - 1] = 1.0
        fd3[counts3 > day_thresh - 1] = 1.0
        fd4[counts4 > day_thresh - 1] = 1.0

        ## showing that final timestep is ignored
        slc = slice(-day_thresh-1, -1) ## [start - 20, start - 1]
        fd1m = np.all(v11_05d[slc] < v11_20d[slc], axis=0)
        assert np.all(np.isclose(fd1.astype(np.uint8), fd1m.astype(np.uint8)))
    else:
        slc = slice(-day_thresh, None) ## [start - 19, start]
        fd1 = np.all(v11_05d[slc] < v11_20d[slc], axis=0)
        fd2 = np.all(v11_05d[slc] < v11_20d[slc], axis=0)
        fd3 = np.all(v11_05d[slc] < v11_20d[slc], axis=0)
        fd4 = np.all(v11_05d[slc] < v11_20d[slc], axis=0)

    return fd1, fd2, fd3, fd4


def _rolling_mean_np(arr, window):
    cs = np.cumsum(arr, axis=0)
    return (cs[window:] - cs[:-window]) / window


def _draw_grid_lines(ax, proj, geo, lon_inc, lat_inc, lw=0.5, ls="-", label=False, fontsize=12):
    """
    : Draws lat/lon grid lines in geographical plots.
    """

    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=lw,
                      color='black', alpha=0.3, linestyle=ls)
    gl.xlabels_top = False

    # Determine longitudes
    # Set starting lon to be multiple of lon_inc \
    # (ensuring starting value is lower than LLlon)
    lon = int((geo[0]-lon_inc/2.) / lon_inc) * lon_inc
    lonList = [lon]
    while lon <= geo[1]:
        lon += lon_inc
        # Modify Longitude labels for Dateline crossing (put pos. AND neg. 180,
        # so gridlines will be drawn)
        if lon == 180:
            lonList.append(180.0)
            lonList.append(-180.0)
        elif lon > 180:
            lonList.append(lon - 360.0)
        else:
            lonList.append(lon)
    gl.xlabels_top = False
    if not label:
        gl.xlabels_bottom = False
    gl.xlocator = mticker.FixedLocator(lonList)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {'color':'black', 'size':fontsize}

    # Determine latitudes
    # Set starting lat to be multiple of lat_inc
    # (ensuring starting value is lower than LLlat)
    lat = int((geo[2]-lat_inc/2.) / lat_inc) * lat_inc
    latList = [lat]
    while lat <= geo[3]:
        lat += lat_inc
        latList.append(lat)
    gl.ylabels_right = False
    if not label:
        gl.ylabels_left = False
    gl.ylocator = mticker.FixedLocator(latList)
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'color':'black', 'size':fontsize}


def _find_latlon_inc(geo):
    # Example min/max bounds
    lon_min, lon_max = geo[0], geo[1]
    lat_min, lat_max = geo[2], geo[3]

    # Target number of labels
    target_lat_labels = 6
    target_lon_labels = 8

    # Compute approximate step size (dlat, dlon)
    dlat = (lat_max - lat_min) / (target_lat_labels - 1)
    dlon = (lon_max - lon_min) / (target_lon_labels - 1)

    # Round step sizes to nearest "nice" integer
    dlat = max(1, round(dlat))   # Ensure at least 1°
    dlon = max(1, round(dlon))
    return dlat, dlon

def plot_geo_ints(int_data, lat, lon, shapes=None,
    geo_bounds=None, latlon_ticks=True,
    int_labels=None, fig_path=None, cbar_ticks=False, colors=None,
    show=False, plot_spec={}):
    """
    Plots a map with pixels colored according to a 2D array of integer values.

    :@param int_data: 2D numpy array of integer values to be visualized
    :@param latitudes: 1D array of latitudes corresponding to rows in `data`
    :@param longitudes: 1D array of longitudes corresponding to columns in`data`
    :@param colors: list or dict mapping indeces present in int_data to
        matplotlib-valid colors
    """
    ps = {
        "xlabel":"", "ylabel":"", "title":"", "dpi":200, "norm":None,
        "figsize":(12,12), "legend_ncols":1, "line_opacity":1, "cmap":"hsv",
        "label_size":14, "title_size":20, "shape_params":{"edgecolor":"black"},
        "cartopy_feats":["land", "borders", "states"],
        }
    ps.update(plot_spec)
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    if shapes:
        ax.add_geometries(
                shapes, ccrs.PlateCarree(), **ps.get("shape_params"))

    if "land" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.LAND,
                #linestyle=ps.get("border_style", "-"),
                #linewidth=ps.get("border_linewidth", 2),
                #edgecolor=ps.get("border_color", "black"),
                )
    if "borders" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.BORDERS,
                linestyle=ps.get("border_style", "-"),
                linewidth=ps.get("border_linewidth", 2),
                edgecolor=ps.get("border_color", "black"),
                )
    if "states" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.STATES,
                linestyle=ps.get("border_style", "-"),
                linewidth=ps.get("border_linewidth", 2),
                edgecolor=ps.get("border_color", "black"),
                )
    if geo_bounds is None:
        geo_bounds = [np.amin(lon), np.amax(lon), np.amin(lat), np.amax(lat)]
    ax.set_extent(geo_bounds, crs=ccrs.PlateCarree())

    m_invalid = ~np.isfinite(int_data)
    int_data[m_invalid] = int_data[~m_invalid][0]
    int_data = int_data.astype(int)

    ## assign each unique integer to an index
    unq_ints = np.unique(int_data)
    val_to_ix = {v:ix for ix,v in enumerate(unq_ints)}
    if colors is None:
        ref_cmap = plt.get_cmap(ps.get("cmap", "tab20"), unq_ints.size)
        cmap = ListedColormap([ref_cmap(i) for i in range(unq_ints.size)])
    else:
        cmap = ListedColormap([colors[v] for v in unq_ints])
    if int_labels is None:
        ix_labels = list(unq_ints)
    else:
        ix_labels = [int_labels[v] for v in unq_ints]
    ix_data = np.vectorize(val_to_ix.get)(int_data).astype(float)
    ix_data[m_invalid] = np.nan

    im = ax.imshow(
            ix_data,
            origin=ps.get("origin", "upper"),
            cmap=cmap,
            extent=geo_bounds,
            interpolation=ps.get("interpolation")
            )

    if latlon_ticks:
        lonmin,lonmax,latmin,latmax = geo_bounds
        frq = ps.get("tick_frequency", 1)
        ax.set_yticks(np.linspace(latmin,latmax,ix_data.shape[0])[::frq],
                crs=ccrs.PlateCarree())
        ax.set_xticks(np.linspace(lonmin,lonmax,ix_data.shape[1])[::frq],
                crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(rotation=ps.get("tick_rotation", 0))
    cbar = plt.colorbar(
            im, ax=ax,
            orientation=ps.get("cbar_orient", "vertical"),
            pad=ps.get("cbar_pad", 0.05),
            shrink=ps.get("cbar_shrink", 1.)
            )

    ## make a scale that centers ticks on their color bar increments
    if cbar_ticks:
        nunq = unq_ints.size
        ticks = np.linspace(0, nunq-1, nunq*2+1)[1::2]
        #ticks = np.array(list(range(nunq))) * (nunq-1)/nunq + .5
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(rotation=ps.get("cbar_tick_rotation", 0))
        cbar.set_ticklabels(ix_labels)
        cbar.ax.tick_params(labelsize=ps.get("cbar_fontsize", 14))

    cbar.set_label(ps.get("cbar_label"))
    ax.set_title(ps.get("title", ""), fontsize=ps.get("title_fontsize", 18))
    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(Path(fig_path).as_posix(), bbox_inches="tight",
                dpi=ps.get("dpi"))
    if show:
        plt.show()
    plt.close()
    return

def get_bounding_latlon_slice(lat, lon, lat_bounds=None, lon_bounds=None):
    """
    Calculate minimum spanning pixel index bounds for the provided latitude
    and longitude arrays given optional coordinate constraints

    :@param lat: 2d array of latitude values
    :@param lon: 2d array of longitude values
    :@param lat_bounds: 2-tuple (min, max) bounds to apply to the domain
    :@param lon_bounds: 2-tuple (min, max) bounds to apply to the domain

    :@return: 2-tuple of slices (slice_y, slice_x) extracting a rectangle
        around the valid domain (inclusive wrt the provided bounds).
    """
    assert lat.shape==lon.shape and lon.ndim==2
    ## establish the bounding box for analysis
    if lat_bounds is None:
        ymin,ymax = np.amin(lat),np.amax(lat)
    else:
        ymin,ymax = lat_bounds
    if lon_bounds is None:
        xmin,xmax = np.amin(lon),np.amax(lon)
    else:
        xmin,xmax = lon_bounds
    ## determine the 2d subgrid bounding box given the provided bounds
    m_valid = (lat >= ymin) & (lat <= ymax) & (lon >= xmin) & (lon <= xmax)
    assert np.any(m_valid), \
        "provided latlon bounds are out of range of the provided coord arrays"
    m_valid_y = np.any(m_valid, axis=1)
    m_valid_x = np.any(m_valid, axis=0)
    slcy = slice(np.argmax(m_valid_y),
            m_valid_y.size - np.argmax(m_valid_y[::-1]))
    slcx = slice(np.argmax(m_valid_x),
            m_valid_x.size - np.argmax(m_valid_x[::-1]))
    return slcy,slcx

def get_poly_raster(latitudes, longitudes, shapefile:Path,
    lat_bounds=None, lon_bounds=None, shapefile_columns:list=None,
    return_subgrid_slices=False, debug=False):
    """
    Given latitude and longitude coordinate arrays and a shapefile, return
    an integer array assigning each pixel to the polygon that contains it,
    maintaining metadata about the polygons from the shapefile.
    Optionally provide latitude and longitude bounds to subset the grid.

    :@param lat: 2d array of latitude values in the domain
    :@param lon: 2d array of longitude values in the domain
    :@param shapefile: Shapefile containing polygons within the provided
        lat/lon domain.
    :@param lat_bounds: optional (min,max) bounds for returned array
    :@param lon_bounds: optional (min,max) bounds for returned array
    :@param shapefile_columns: List of strings matching the names of auxiliary
        columns in the geojson to return alongside the raster.
    :@param return_subgrid_slices: Boolean; if True, also returns a 2-tuple of
        slices (yslice, xslice) that extract the subgrid of the provided lat
        and lon arrays conforming the the provided bounds

    :@return: 2-tuple (poly_ints, metadata). poly_ints is an array of integer
        values shaped identically to the latitude and longitude arrays, such
        that the integers indicate which polygon each pixel falls within.
        metadata is a list of dicts that is equal in length to the number of
        unique values in poly_ints, such that poly_ints's values provide
        the index of the corresponding polygon dictionary. Each dict contains
        at least one field "poly_idx" providing the integer of that polygon
        with respect to the original shapefile, but may contain additional
        fields as specified by shapefile_columns. If return_subgrid_slices is
        True, returns 3-tuple like:
        (poly_ints:np.array, metadata:list, (yslice:slice, xslice:slice))
    """
    if debug:
        print(f"{time.perf_counter():.3f} Reading shapefile ")
    lat,lon = latitudes,longitudes
    ## extract the polygons from the shapefile
    gdf = gpd.read_file(shapefile)

    colkeys = []
    if not shapefile_columns is None:
        for k in shapefile_columns:
            assert k in gdf.keys(), f"Not found in shapefile columns: {k}"
            colkeys.append(k)

    ## retain only polygongs that intersect the overall lat/lon bounding box
    polys = []

    ## establish the bounding box for analysis
    if lat_bounds is None:
        ymin,ymax = np.amin(lat),np.amax(lat)
    else:
        ymin,ymax = lat_bounds
    if lon_bounds is None:
        xmin,xmax = np.amin(lon),np.amax(lon)
    else:
        xmin,xmax = lon_bounds
    bbox = Polygon([
        (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

    ## determine the 2d subgrid bounding box given the provided bounds
    slcy,slcx = get_bounding_latlon_slice(lat, lon, lat_bounds, lon_bounds)

    ## subset the coordinate arrays
    lat = lat[slcy,slcx]
    lon = lon[slcy,slcx]

    ## make a shapely point for each coordinate combination
    flat_lat = lat.ravel()
    flat_lon = lon.ravel()
    points = [Point(x, y) for x, y in zip(flat_lon, flat_lat)]

    ## Subset the polygons to only those which intersecet the bounding box
    poly_ixs,poly_ids,polygons = zip(*[
        (i,id(p),p)
        for i,p in enumerate(gdf.geometry.values) if p.intersects(bbox)
        ])
    id_to_ix = dict(zip(poly_ids,poly_ixs))

    if debug:
        print(f"{time.perf_counter():.3f} Initializing STR Tree ")
    ## make an STR tree of the polygons so that it's efficient to rule out
    ## inclusion of pixels that are strictly outside the minimum bounding
    ## rectangle. See linked document:
    ## https://ia600709.us.archive.org/13/items/nasa_techdoc_19970016975/19970016975.pdf
    tree = STRtree(polygons)

    if debug:
        print(f"{time.perf_counter():.3f} Grouping by polygons ")
    ## For each of the points, see if it is in any of the polygon's MBR
    ## by querying the STR tree. Then do a refined check to see which of the
    ## polygons actually contain it.
    poly_raster = np.full(len(points), -1, dtype=int)
    for rix,pt in enumerate(points):
        if int(shapely.__version__[0])==1:
            ## tree only contains polygons from subset so must convert to the
            ## polygon indeces wrt the shapefile ordering if version 1
            cand_polys = tree.query(pt)
            cand_pixs = [id_to_ix[id(p)] for p in cand_polys]
        else:
            ## otherwise query returns the indeces wrt the input polys
            cand_poly_subset_ixs = tree.query(pt)
            cand_polys = [polygons[ix] for ix in cand_poly_subset_ixs]
            cand_pixs = [poly_ixs[ix] for ix in cand_poly_subset_ixs]
        ## use the new polygon indeces, not the ones from the shapefile.
        ## the original shapefile indeces will be returned in the metadata
        #for pix,poly in enumerate(cand_polys):
        for pix,poly in zip(cand_pixs,cand_polys):
            if poly.contains(pt):
                poly_raster[rix] = pix
                break

    ## extract the requested auxiliary column data from the polygons, and
    ## convert the int values from the original polygon indeces to contiguous
    ## values starting at 0, with -1 still representing masked values
    unq_pixs = np.unique(poly_raster)
    if -1 in unq_pixs:
        unq_pixs = np.delete(unq_pixs, 0) ## -1 should always be 0 index
    metadata = [{"poly_idx":pix, **{k:gdf[k][pix] for k in colkeys}}
        for i,pix in enumerate(unq_pixs)]
    val_to_ix = {v:ix for ix,v in enumerate(unq_pixs)}
    val_to_ix[-1] = -1
    poly_raster = np.vectorize(val_to_ix.get)(poly_raster)

    if return_subgrid_slices:
        return poly_raster.reshape(lat.shape),metadata,(slcy,slcx)
    return poly_raster.reshape(lat.shape),metadata


def apply_by_polygon(dataset, poly_int_raster, agg_func,
        dtype=np.float32, poly_oob_value=-1, output_oob_value=np.nan):
    """
    Given a (Y,X) dataset and a (Y,X) polygon integer raster array
    (returned by get_poly_raster), apply a function mapping all pixels within
    that polygon to a single value, and return an array with the same shape as
    the input dataset, but with every pixel within each polygon having the
    value returned by the function.
    """
    out = np.full(dataset.shape, output_oob_value, dtype=dtype)
    for pix in np.unique(poly_int_raster):
        if pix==poly_oob_value:
            continue
        m_pix = poly_int_raster==pix
        tmp = np.apply_along_axis(
                func1d=agg_func, axis=0, arr=dataset[m_pix]
                )
        out[m_pix] = tmp
    return out

def make_countyfd_plots(lons,lats,fd11,geo,title1,fname1,region,smvi_thresh):
    poly_raster_path = Path(POLY_RASTER_DIR).joinpath(
            f"poly-raster_counties_{region}.pkl")
    assert lons.ndim==1 and lats.ndim==1
    lons = ((lons + 180) % 360) - 180
    ## coming from sportlis_smvi.py, the lat axis counts from low to high.
    ## flip the axis to be consistent with with get_poly_raster's convention
    fd11 = fd11[::-1]
    lat2d = np.stack([lats for i in range(lons.shape[0])],axis=1)[::-1]
    lon2d = np.stack([lons for i in range(lats.shape[0])],axis=0)
    if not poly_raster_path.exists():
        print(f"Needed polygon raster doesn't exist at {poly_raster_path}.",
                "Generating it now.")
        assert Path(POLY_RASTER_DIR).exists(),POLY_RASTER_DIR
        pir,metadata,sub_slice = get_poly_raster(
                latitudes=lat2d,
                longitudes=lon2d,
                shapefile=county_shp,
                lat_bounds=geo[2:],
                lon_bounds=geo[:2],
                shapefile_columns=[
                    "STATE","CWA","COUNTYNAME","FIPS","TIME_ZONE",
                    "FE_AREA","LON","LAT","Shape_Area"],
                return_subgrid_slices=True,
                debug=True,
                )
        yslc,xslc = sub_slice
        latlon = (lat2d[yslc,xslc],lon2d[yslc,xslc])
        pickle.dump((pir,metadata,sub_slice,latlon),
                poly_raster_path.open("wb"))
        print(f"Created new poly raster file at {poly_raster_path.as_posix()}")
    else:
        pir,metadata,sub_slice,_ = pickle.load(
                poly_raster_path.open("rb"))
        print(f"Loaded poly raster path from {poly_raster_path.as_posix()}")
    fd11 = fd11[sub_slice[0], sub_slice[1]]-1
    poly_smvi = apply_by_polygon(
        dataset=np.where(fd11>=0,fd11,np.nan),
        poly_int_raster=pir,
        agg_func=lambda x:(np.average(x)>smvi_thresh).astype(int),
        output_oob_value=np.nan,
        )
    poly_smvi = np.where(poly_smvi>=0, poly_smvi+1, 0)
    geoms = tuple(counties.geometries())
    plot_geo_ints(
        int_data=poly_smvi,
        lat=lat2d[sub_slice[0], sub_slice[1]],
        lon=lon2d[sub_slice[0], sub_slice[1]],
        fig_path=fname1,
        latlon_ticks=False,
        shapes=[geoms[md["poly_idx"]] for md in metadata],
        cbar_ticks=True,
        plot_spec={
            "cbar_pad":0.02,
            "cbar_orient":"horizontal",
            "cbar_shrink":.8,
            "cbar_fontsize":14,
            "tick_frequency":12,
            "tick_rotation":45,
            "title":f"Counties with >{smvi_thresh*100}% Flash Drought Area",
            "tile_fontsize":18,
            "interpolation":"none",
            "shape_params":{
                "edgecolor":"silver",
                "facecolor":"none",
                "alpha":.85,
                },
            },
        colors=["#3D74B6", "#FBF5DE", "#DC3C22"],
        )
