import numpy as np
import pygrib
import imageio.v2 as imageio
import pickle as pkl
from pathlib import Path
from multiprocessing import Pool
from pprint import pprint
from datetime import datetime
from shapely.strtree import STRtree

from plotting import plot_geo_scalar,plot_geo_ints
from get_sportlis_smvi import SMVIConfig,get_sportlis_smvi
from get_poly_raster import get_poly_raster

## labels corresponding to each sportlis_HIST record (extracted with wgrib)
rlabels_hist = [
    "lhtfl", "shtfl", "gflux", "evp", "ssrun", "bgrun", "avsft", "albdo",
    "weasd", "soilm-10", "soilm-40", "soilm-100", "soilm-200", "tsoil-10",
    "tsoil-40", "tsoil-100", "tsoil-200", "mstav", "pevap", "cnwat",
    "wind", "tmp", "spfh", "pres", "dswrf", "dlwrf", "land", "vgtyp",
    "sotyp", "dist", "lai", "veg", "mstav-10", "mstav-40", "mstav-100",
    "mstav-200", "apcp"]
## labels corresponding to each sportlis_vsm_percentile record
rlabels_pct = ["soilm-10", "soilm-40", "soilm-100", "soilm-200"]

if __name__=="__main__":
    data_dir = Path("data")
    fig_dir = Path("figures/daily")
    gif_dir = Path("figures/gifs")
    sportlis_dir = data_dir.joinpath("sportlis-2016")
    shapefile = data_dir.joinpath("shapefiles/c_15au13.shp")
    latlon = np.load(data_dir.joinpath("sportlis_latlon.npy"))
    lat,lon = latlon[...,0],latlon[...,1]

    ## configure geographic and temporal ranges, and data features for which
    ## to calculate daily county-wise SMVI
    #lat_bounds,lon_bounds,bbox_name = (32,38),(-87,-79),"EastTN"
    lat_bounds,lon_bounds,bbox_name = (24.5,31.5),(-88,-80),"Florida"

    start_time = datetime(2016,9,30)
    end_time = datetime(2016,12,31)
    ## labels of soil layers
    soilm_labels = ["soilm-10", "soilm-40", "soilm-100", "soilm-200"]
    layer_depths = [.1, .3, .6, 1.]
    ## If True, re-calculates raster rather than using stored
    new_poly_raster = False
    ## If True, re-calculates SMVI rather than using stored
    new_smvi = False
    gif_fps = 8

    ## define output paths for the intermediate data files
    ts0 = start_time.strftime("%Y%m%d")
    tsf = end_time.strftime("%Y%m%d")
    smvi_path = data_dir.joinpath(f"smvi/smvi_{bbox_name}_{ts0}_{tsf}.npy")
    poly_raster_path = data_dir.joinpath(f"poly/poly-raster_{bbox_name}.pkl")

    ## generate a raster file assigning each pixel to a county polygon
    if new_poly_raster:
        pir,metadata,sub_slice = get_poly_raster(
                latitudes=lat,
                longitudes=lon,
                shapefile=shapefile,
                lat_bounds=lat_bounds,
                lon_bounds=lon_bounds,
                shapefile_columns=[
                    "STATE","CWA","COUNTYNAME","FIPS","TIME_ZONE",
                    "FE_AREA","LON","LAT","Shape_Area"],
                return_subgrid_slices=True,
                )
        latlon = (lat[*sub_slice],lon[*sub_slice])
        pkl.dump((pir,metadata,sub_slice,latlon), poly_raster_path.open("wb"))
    ## test plot for polygon index raster
    #plot_geo_ints(int_data=pir, lat=lat, lon=lon, int_labels=None, show=True)

    ## calculate SMVI over the specified date range, using the same bounds as
    ## the county polygon raster
    if new_smvi:
        ## indices of soil layers wrt the LIS_HIST
        hist_soilm_record_idxs = [rlabels_hist.index(l) for l in soilm_labels]
        ## labels of corresponding soil layers in the percentile file
        pct_soilm_record_idxs = list(range(4))

        smvi,dates = get_sportlis_smvi(
            hist_file_dir=sportlis_dir,
            percentile_file_dir=sportlis_dir,
            hist_record_indices=hist_soilm_record_idxs,
            percentile_record_indices=pct_soilm_record_idxs,
            layer_depths=layer_depths,
            start_time=start_time,
            end_time=end_time,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            smvi_config=SMVIConfig(),
            nworkers=4,
            ngroups=10,
            )
        pkl.dump((smvi,dates,soilm_labels), smvi_path.open("wb"))

    ## load the stored pkl files and plot
    pir,metadata,sub_slice,(lat,lon) = pkl.load(poly_raster_path.open("rb"))
    smvi,dates,soilm_labels = pkl.load(smvi_path.open("rb"))
    smvi_frac = np.full(smvi.shape, np.nan)
    for pix in np.unique(pir): ## iterating over polygons
        if pix==-1:
            continue
        m_pix = pir==pix ## mask of this polygon
        npx = np.count_nonzero(m_pix) ## pixels in this polygon
        ## fraction of pixels in this polygon that have volatility
        fsmvi = np.count_nonzero(smvi[:,m_pix,:]==1, axis=1) / npx
        smvi_frac[:,m_pix,:] = fsmvi[:,np.newaxis]

    plotted_files = {}
    for fix,fstr in enumerate(soilm_labels):
        plotted_files[fstr] = []
        for tix,dt in enumerate(dates):
            tstr = dt.strftime("%Y%m%d")
            tstr2 = dt.strftime("%Y-%m-%d")
            fig_path = fig_dir.joinpath(
                    f"smvi_frac_{bbox_name}_{tstr}_{fstr}.png")
            plot_geo_scalar(
                data=smvi_frac[tix,:,:,fix],
                latitude=lat,
                longitude=lon,
                plot_spec={
                    "cmap":"RdYlGn_r",
                    "title":f"{fstr} SMVI % per county ({tstr2})",
                    "figsize":(24,16),
                    "cbar_shrink":.9,
                    "vmin":0,
                    "vmax":1,
                    },
                latlon_ticks=False,
                show=False,
                fig_path=fig_path,
                )
            plotted_files[fstr].append(fig_path)
            print(f"Generated {fig_path.as_posix()}")

    for fix,fstr in enumerate(plotted_files.keys()):
        ts0 = dates[0].strftime("%Y%m%d")
        tsf = dates[-1].strftime("%Y%m%d")
        gif_file_path = gif_dir.joinpath(
            f"smvi_frac_{bbox_name}_{ts0}-{tsf}_{fstr}.gif")
        images = []
        for p in plotted_files[fstr]:
            images.append(imageio.imread(p))
            imageio.mimsave(gif_file_path, images, duration=1/gif_fps)
