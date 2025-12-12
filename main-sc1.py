#!/raid2/sport/people/casejl/python/mamba/install/envs/liw2/bin/python
import numpy as np
import geopandas as gpd
import pickle as pkl
import argparse
import re
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime

from plotting import mp_plot_binary_smvi,mp_plot_percentile_and_smvi
from get_sportlis_smvi import SMVIConfig,get_sportlis_smvi,load_sportlis_gribs
from get_poly_raster import get_poly_raster,apply_by_polygon

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
dow_options = ["mon", "tue", "wed", "thu", "fri", "sat", "sun", "all"]

def build_parser():
    """ CLI parser """
    def valid_domain(value):
        """Validate domain"""
        if not value in geog.keys():
            raise ValueError(
                f"domain {value} must be one of {list(geog.keys())}")

    def valid_date(value):
        """Validate YYYYmmdd format."""
        if not re.fullmatch(r"\d{8}", value):
            raise ValueError(
                f"Invalid date '{value}'. Expected format: YYYYmmdd")
        return value

    def valid_day_of_week(value):
        """Validate day of week string """
        if not value.lower() in dow_options:
            raise ValueError(
                f"day_of_week must be one of: {dow_options}")
        return value

    parser = argparse.ArgumentParser(
        description="Specify a region and time range for SPoRT-LIS SMVI")
    parser.add_argument(
        "domain", type=str,
        help="Domain to process (string).")
    parser.add_argument(
        "start_day", type=valid_date,
        help="Start day in YYYYmmdd format.")
    parser.add_argument(
        "end_day", nargs="?", type=valid_date,
        help="Optional end day in YYYYmmdd format.")
    parser.add_argument(
        "-d", "--day_of_week", type=valid_day_of_week, default="ALL",
        help=f"String depicting day of week. Options: {dow_options}")
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.3,
        help="SMVI threshold pixel ratio for activation (default: 0.3).")
    return parser

if __name__=="__main__":
    sportlis_parent_dir = Path(
            "/raid2/sport/people/casejl/LIS7/OUTPUT/conus3km/SURFACEMODEL")
    fig_dir = Path("figures/daily")

    ## longitude storage formats are inconsistent between some grids; just
    ## supply a static save file of the full sportlis latlon domain.
    latlon_file = Path("data/sportlis_latlon.npy")

    ## parent directories where each file type are located
    pctl_parent_dir = sportlis_parent_dir.joinpath(
            "grid_percentile/statsgo-orig-nldas2")
    hist_parent_dir = sportlis_parent_dir

    ## shapefile defining polygons pixels, and a unique name for it
    shapefile = Path("data/shapefiles/c_15au13.shp")
    poly_name = "counties"

    ## directory where pkls containing poly raster domains are stored
    poly_raster_dir = Path("data/poly")
    new_poly_raster = False ## if True, always re-generates poly rasters

    ## verify that required files/directories exist
    assert sportlis_parent_dir.exists(),sportlis_parent_dir
    assert latlon_file.exists(),latlon_file
    assert pctl_parent_dir.exists(),pctl_parent_dir
    assert hist_parent_dir.exists(),hist_parent_dir
    assert shapefile.exists(),shapefile
    assert poly_raster_dir.exists(),poly_raster_dir

    ## sc1 file naming scheme
    percentile_file_pattern="{yyyy}/vsm_percentile_{yyyymmdd}.grb2"
    hist_file_pattern="{yyyymm}/LIS_HIST_{yyyymmdd}0000.d01.grb2"

    ## labels of soil layers
    soilm_labels = ["soilm-10", "soilm-40", "soilm-100", "soilm-200"]
    layer_depths = [.1, .3, .6, 1.]

    ## plotting options for smvi
    plot_binary_smvi = True
    plot_percentile_and_smvi = False

    ## number of concurrent workers and number of subsets (groups) to split the
    ## time series into. More groups need more memory, but fewer disc reads.
    nworkers = 12
    ngroups = 12
    debug = True

    """   -----( end of normal configuration )-----   """

    ## parse cli arguments
    args = build_parser().parse_args()
    bbox_name = args.domain
    assert bbox_name in geog.keys(), \
            f"domain '{bbox_name}' must be one of {list(geog.keys())}"
    lat_bounds = geog[bbox_name][2:]
    lon_bounds = geog[bbox_name][:2]
    start_time = datetime.strptime(args.start_day, "%Y%m%d")
    if args.end_day is None:
        end_time = start_time
    else:
        end_time = datetime.strptime(args.end_day, "%Y%m%d")
    ## percentage threshold of SMVI positive pixels in a polygon in order for
    ## that polygon to be considered "active" in binary plots
    smvi_thresh = args.threshold
    ## optionally provide a exclusive day of the week to plot for weekly
    if args.day_of_week.lower() == "all":
        plot_day_of_week = None
    else:
        plot_day_of_week = dow_options.index(args.day_of_week.lower())

    ## extract the geographic coords from the static file
    latlon = np.load(latlon_file)
    lat,lon = latlon[...,0],latlon[...,1]

    ## define output paths for the intermediate data files
    ts0 = start_time.strftime("%Y%m%d")
    tsf = end_time.strftime("%Y%m%d")

    ## if a new poly raster is needed or requested, generate it
    poly_raster_path = poly_raster_dir.joinpath(
            f"poly-raster_{poly_name}_{bbox_name}.pkl")
    if new_poly_raster or not poly_raster_path.exists():
        ## generate a raster file assigning each pixel to a county polygon
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
                debug=debug,
                )
        yslc,xslc = sub_slice
        latlon = (lat[yslc,xslc],lon[yslc,xslc])
        pkl.dump((pir,metadata,sub_slice,latlon), poly_raster_path.open("wb"))
    else:
        assert poly_raster_path.exists(),poly_raster_path
        pir,metadata,sub_slice,(lat,lon) = pkl.load(
                poly_raster_path.open("rb"))

    ## indices of soil layers wrt the LIS_HIST file records
    hist_soilm_record_idxs = [rlabels_hist.index(l) for l in soilm_labels]
    ## indices of soil layers wrt the percentile file records
    pct_soilm_record_idxs = [rlabels_pct.index(l) for l in soilm_labels]

    plotted_files = {}
    ## calculate SMVI over the specified date range, using the same bounds as
    ## the county polygon raster
    smvi,dates,(hist_files,pct_files) = get_sportlis_smvi(
        hist_file_dir=hist_parent_dir,
        percentile_file_dir=pctl_parent_dir,
        hist_record_indices=hist_soilm_record_idxs,
        percentile_record_indices=pct_soilm_record_idxs,
        layer_depths=layer_depths,
        start_time=start_time,
        end_time=end_time,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        smvi_config=SMVIConfig(),
        nworkers=nworkers,
        ngroups=ngroups,
        latitudes=lat,
        longitudes=lon,
        percentile_file_pattern=percentile_file_pattern,
        hist_file_pattern=hist_file_pattern,
        return_source_files=True,
        debug=debug,
        )

    ## get the relevant counties from the indeces in the metadata
    gdf = gpd.read_file(shapefile)
    polys = [gdf["geometry"][md["poly_idx"]] for md in metadata]

    if plot_percentile_and_smvi:
        if debug:
            print(f"loading {len(pct_files)} percentile gribs")
        pct_data,_ = load_sportlis_gribs(
                grib_paths=pct_files,
                record_indices=pct_soilm_record_idxs,
                slice_bounds=sub_slice,
                return_original_shape=True,
                mask_value=9999.,
                debug=debug,
                )

        args = [{
            "percentile_data":pct_data[tix,:,:,fix],
            "smvi_data":(smvi[tix,:,:,fix]==1),
            "lat":lat,
            "lon":lon,
            "fstr":fstr,
            "fig_path":fig_dir.joinpath(
                f"smvi_pixelwise-percentile_{bbox_name}_" + \
                        f"{dt.strftime('%Y%m%d')}_{fstr}.png"),
            "polys":polys,
            "smvi_thresh":smvi_thresh,
            "date":dt,
            }
            for tix,dt in enumerate(dates)
            for fix,fstr in enumerate(soilm_labels)
            if plot_day_of_week is None or dt.weekday()==plot_day_of_week
            ]
        print([dt.weekday() for dt in dates])
        print(f"plot_day_of_week={plot_day_of_week}")
        print(len(args))
        with Pool(nworkers) as pool:
            for p in pool.imap_unordered(mp_plot_percentile_and_smvi, args):
                print(f"Generated {p.as_posix()}")

    if plot_binary_smvi:
        ## apply a function to each polygon returning 1 when the fraction of
        ## pixels in the polygon > smvi_thresh, 0 otherwise.
        poly_smvi = apply_by_polygon(
            dataset=smvi,
            poly_int_raster=pir,
            agg_func=lambda x:(np.average(x)>smvi_thresh).astype(int),
            output_oob_value=np.nan,
            )
        ## modify the ints so that 0 is out of bounds, 1 is in-bounds but SMVI
        ## below threshold, and 2 represents polygons exceeding the threshold
        poly_smvi = np.where(poly_smvi>=0, poly_smvi+1, 0)

        args = [{
            "int_data":poly_smvi[tix,:,:,fix],
            "lat":lat,
            "lon":lon,
            "fstr":fstr,
            "fig_path":fig_dir.joinpath(
                f"smvi_binary_{bbox_name}_{poly_name}_" + \
                        f"{dt.strftime('%Y%m%d')}_{fstr}.png"),
            "polys":polys,
            "smvi_thresh":smvi_thresh,
            "date":dt,
            }
            for tix,dt in enumerate(dates)
            for fix,fstr in enumerate(soilm_labels)
            if plot_day_of_week is None or dt.weekday()==plot_day_of_week
            ]

        with Pool(nworkers) as pool:
            for p in pool.imap_unordered(mp_plot_binary_smvi, args):
                print(f"Generated {p.as_posix()}")
