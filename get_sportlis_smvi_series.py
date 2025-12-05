import numpy as np
import geopandas as gpd
import pygrib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from numpy.lib.stride_tricks import sliding_window_view
from dataclasses import dataclass
from multiprocessing import Pool
from pprint import pprint
from datetime import datetime,timedelta
from cartopy.mpl.ticker import LatitudeFormatter,LongitudeFormatter
from matplotlib.colors import ListedColormap
from pathlib import Path
from shapely.geometry import Point,Polygon
from shapely.strtree import STRtree

@dataclass
class SMVIConfig:
    """
    Configuration object defining a set of SMVI rules.

    SMVI is a boolean function such that it returns True when:

     - Daily big window average is greater than small window average for a
       number of days.
     - The climatological percentile on the last day is below a cutoff
    """
    big_window_size:int=20
    small_window_size:int=5
    drying_days:int=20
    last_day_percentile_cutoff:float=20

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

def get_sportlis_smvi_series(
        hist_file_dir:Path, percentile_file_dir:Path,
        hist_record_indices:tuple, percentile_record_indices:tuple,
        start_time:datetime, end_time:datetime,
        lat_bounds:tuple, lon_bounds:tuple,
        smvi_config:SMVIConfig=None,
        integrate_layers=True, layer_depths=None,
        nworkers=1, ngroups=4,
        ):
    """
    :@param hist_file_dir: Directory containing SPoRT LIS model state files
        conforming to the naming pattern sportlis_HIST_{yyyymmdd}0000_d01.grb
    :@param percentile_file_dir: Directory containing SPoRT LIS percentile
        files conforming to name: sportlis_vsm_percentile_{yyyymmdd}.grb2
    :@param hist_record_indices: Indeces of the grib records within the HIST
        files corresponding to each of the soil layers considered for SMVI.
        IMPORTANT: grib records start at 1 by convention, but in this case
        the first record is indexed 0 (as is the case in python).
    :@param percentile_record_indices: Indeces of the grib records within the
        percentile files corresponding to each of the soil layers.
        IMPORTANT: grib records start at 1 by convention, but in this case
        the first record is indexed 0 (as is the case in python).
    :@param start_time: Initial time for which SMVI will be reported. 40 days'
        worth of files prior to this day must be present in order to be valid.
    :@param end_time: Final time for which SMVI will be reported.
    :@param lat_bounds: 2-tuple (min, max) subgrid latitude bounds for analysis
    :@param lon_bounds: 2-tuple (min, max) subgrid longitude bounds
    :@param integrate_layers: If True, volumetric soil moisture will be
        accumulated for each layer listed before a given layer according to
        the layer depths (which must be provided if True). In other words,
        the third layer (third index in hist_record_indices) will be the
        weighted sum of layers 1, 2, and 3.
    :@param layer_depths: Relative depths of each layer (arbitrary units)
    :@param nworkers: Number of workers used in parallel to calculate SMVI.
        Each worker handles a single group at a time.
    :@param ngroups: Number of time segments to subdivide the data into. The
        more groups there are, the less data is loaded into memory at once,
        but the higher the probability that a file will need to be read from
        the disc multiple times.

    :@return: Array of SMVI values shaped like (N,Y,X,F) for N days between
        start_time and end_time inclusively, and F soil layers
    """
    fpat_hist = "sportlis_HIST_{yyyymmdd}0000_d01.grb" ## only 0z files
    fpat_pct = "sportlis_vsm_percentile_{yyyymmdd}.grb2"

    ## check the validity of layer-defininig arguments
    assert len(hist_record_indices)==len(percentile_record_indices)
    if integrate_layers:
        assert not layer_depths is None, \
            "layer depths must be provided since integrate_layers=True"
        assert len(layer_depths) == len(hist_record_indices)

    ## just use the defaults if not provided
    if smvi_config is None:
        smvi_config = SMVIConfig()
    assert smvi_config.big_window_size > smvi_config.small_window_size

    ## build a list of dates needed for each of the file types, taking into
    ## consideration the moving window condition sizes. Note that each time a
    ## moving window is applied the number of elements lost is the size of the
    ## window minus 1
    hist_dates = []
    pct_dates = []
    lag_days = smvi_config.big_window_size + smvi_config.drying_days - 2
    t = start_time - timedelta(days=lag_days)
    while t <= end_time:
        tstr = t.strftime("%Y%m%d")
        hist_dates.append(tstr)
        if t >= start_time:
            pct_dates.append(tstr)
        t += timedelta(days=1)

    ## explicitly calculate the needed file names given the earthdata scheme,
    ## and make sure they exist in the provided directories.
    req_hist = [hist_file_dir.joinpath(fpat_hist.format(yyyymmdd=tstr))
                for tstr in hist_dates]
    req_pct = [percentile_file_dir.joinpath(fpat_pct.format(yyyymmdd=tstr))
                for tstr in pct_dates]
    for p in req_hist + req_pct:
        if not p.exists():
            raise ValueError(f"Missing required file: {p.as_posix()}")

    ## assume the grids are all uniform and extract coordinate arrays
    with pygrib.open(req_hist[0]) as pgf:
        lat,lon = pgf.message(1).latlons()

    ## get the bounding box of the subgrid desired for analysis
    slice_bounds = get_bounding_latlon_slice(lat, lon, lat_bounds, lon_bounds)

    ## break up the time series into groups
    wkr_steps = len(pct_dates) // ngroups
    res_steps = len(pct_dates) % ngroups
    args = [{
        "hist_paths":req_hist[
            i*wkr_steps:(i+1)*wkr_steps+[0,res_steps][i==ngroups-1]+lag_days],
        "pct_paths":req_pct[
            i*wkr_steps:(i+1)*wkr_steps+[0,res_steps][i==ngroups-1]],
        "hist_rixs":hist_record_indices,
        "pct_rixs":percentile_record_indices,
        "layer_depths":layer_depths,
        "slice_bounds":slice_bounds,
        "integrate_layers":integrate_layers,
        "layer_depths":np.asarray(layer_depths),
        "wbig":smvi_config.big_window_size,
        "wsmall":smvi_config.small_window_size,
        "ddays":smvi_config.drying_days,
        "pct_thresh":smvi_config.last_day_percentile_cutoff,
        } for i in range(ngroups)]

    smvi = []
    with Pool(nworkers) as pool:
        for result in pool.imap(_mp_get_smvi, args):
            smvi.append(result)
    return np.concatenate(smvi, axis=0)

def _mp_get_smvi(args):
    return _get_smvi(**args)

def _get_smvi(
    hist_paths, pct_paths, hist_rixs, pct_rixs, layer_depths, slice_bounds,
    integrate_layers, wbig, wsmall, ddays, pct_thresh):
    """
    Internal module method for retrieving SMVI given SPoRT LIS soil state and
    percentile files. You SHOULD NOT use this method directly... it is wrapped
    by get_sportlis_smvi_series, and expects guarunteed daily chronological and
    temporally aligned files to be provided.
    """
    ## read all the state data into memory
    hist = []
    m_valid = None
    for p in hist_paths:
        with pygrib.open(p) as pgf:
            tmp_feats = []
            for rix in hist_rixs:
                pgf.seek(rix)
                x = pgf.readline().values.data[*slice_bounds]
                ## hist files use 9999. as a mask value
                if m_valid is None:
                    m_valid = np.isclose(x, 9999)
                tmp_feats.append(x[m_valid])
            hist.append(np.stack(tmp_feats, axis=-1))

    ## read all the percentile data into memory
    pct = []
    for p in pct_paths:
        with pygrib.open(p) as pgf:
            tmp_feats = []
            for rix in pct_rixs:
                pgf.seek(rix)
                x = pgf.readline().values.data[*slice_bounds]
                tmp_feats.append(x[m_valid])
            pct.append(np.stack(tmp_feats, axis=-1))
    hist = np.stack(hist, axis=0)
    pct = np.stack(pct, axis=0)
    print(f"{hist.shape = } {pct.shape = }")

    ## progressively vertically integrate each layer if requested
    if integrate_layers:
        hist = np.cumsum(hist*layer_depths, axis=-1) / np.cumsum(layer_depths)

    ## declare convolution filters according to the requested window sizes
    fbig = np.ones(wbig)
    fsmall = np.ones(wsmall)

    ## get the moving averages for both the small and big window sizes
    mavg_big = np.apply_along_axis(
            lambda m: np.convolve(m,fbig,mode="valid"),
            axis=0, arr=hist,
            )
    mavg_small = np.apply_along_axis(
            lambda m: np.convolve(m,fsmall,mode="valid"),
            axis=0, arr=hist[wbig-wsmall:],
            )

    ## apply the SMVI condition that the large moving average must be greater
    ## than the small moving average for a particular number of days
    m_drying = np.all(
        sliding_window_view(
            mavg_big > mavg_small,
            window_shape=ddays,
            axis=0),
        axis=-1)
    ## apply the SMVI condition that the final day is below a climatological
    ## percentile threshold
    m_pct = pct < pct_thresh
    ## combine the conditions to calculate SMVi
    return m_drying & m_pct

def get_index_raster_from_polygons(latitudes, longitudes, shapefile:Path,
    lat_bounds=None, lon_bounds=None, shapefile_columns:list=None,
    return_subgrid_slices=False):
    """

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
    print(f"getting points")
    points = np.array([Point(x, y) for x, y in zip(flat_lon, flat_lat)])

    print(f"subsetting polygons")
    ## Subset the polygons to only those which intersecet the bounding box
    poly_ixs,polygons = zip(*[
        (i,p) for i,p in enumerate(gdf.geometry.values) if p.intersects(bbox)
        ])

    ## make an STR tree of the polygons so that it's efficient to rule out
    ## inclusion of pixels that are strictly outside the minimum bounding
    ## rectangle. See linked document:
    ## https://ia600709.us.archive.org/13/items/nasa_techdoc_19970016975/19970016975.pdf
    print(f"got {len(polygons)} polygons; making tree")
    tree = STRtree(polygons)

    ## extract the requested auxiliary column data from the polygons
    metadata = [
        {"poly_idx":pix, **{gdf[k][pix] for k in colkeys}}
        for pix in poly_ixs]

    print(f"searching tree")
    ## For each of the points, see if it is in any of the polygon's MBR
    ## by querying the STR tree. Then do a refined check to see which of the
    ## polygons actually contain it.
    poly_raster = np.full(len(points), -1, dtype=int)
    for rix,pt in enumerate(points):
        ## tree only contains polygons from subset so must convert to the
        ## polygon indeces wrt the shapefile ordering
        cand_poly_subset_ixs = tree.query(pt)
        cand_polys = [polygons[ix] for ix in cand_poly_subset_ixs]
        cand_pixs = [poly_ixs[ix] for ix in cand_poly_subset_ixs]
        ## use the new polygon indeces, not the ones from the shapefile.
        ## the original shapefile indeces will be returned in the metadata
        for pix,poly in enumerate(cand_polys):
            if poly.contains(pt):
                poly_raster[rix] = pix
                break

    if return_subgrid_slices:
        return poly_raster.reshape(lat.shape),metadata,(slcy,slcx)
    return poly_raster.reshape(lat.shape),metadata

def plot_geo_ints(int_data, lat, lon, geo_bounds=None, latlon_ticks=True,
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
    ps = {"xlabel":"", "ylabel":"",
            "title":"", "dpi":80, "norm":None,"figsize":(12,12),
            "legend_ncols":1, "line_opacity":1, "cmap":"hsv",
            "label_size":14, "title_size":20}
    ps.update(plot_spec)
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(
            cfeature.LAND,
            linestyle=ps.get("border_style", "-"),
            linewidth=ps.get("border_linewidth", 2),
            edgecolor=ps.get("border_color", "black"),
            )

    ax.add_feature(
            cfeature.BORDERS,
            linestyle=ps.get("border_style", "-"),
            linewidth=ps.get("border_linewidth", 2),
            edgecolor=ps.get("border_color", "black"),
            )
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
            pad=ps.get("cbar_pad", 0.0),
            shrink=ps.get("cbar_shrink", 1.)
            )

    ## make a scale that centers ticks on their color bar increments
    if cbar_ticks:
        nunq = unq_ints.size
        ticks = np.array(list(range(nunq))) * (nunq-1)/nunq + .5
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(rotation=ps.get("cbar_tick_rotation", 0))
        cbar.set_ticklabels(ix_labels)
        cbar.ax.tick_params(labelsize=ps.get("cbar_fontsize", 14))

    cbar.set_label(ps.get("cbar_label"))
    ax.set_title(ps.get("title", ""), fontsize=ps.get("title_fontsize", 18))
    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)
    if show:
        plt.show()
    plt.close()
    return

if __name__=="__main__":
    data_dir = Path("data")
    sportlis_dir = data_dir.joinpath("sportlis-2016")
    shapefile = data_dir.joinpath("shapefiles/c_15au13.shp")

    lat_bounds = (34.5, 37)
    lon_bounds = (-85, -81)
    start_time = datetime(2016,6,1)
    end_time = datetime(2016,9,30)
    ## labels of soil layers
    soilm_labels = ["soilm-10", "soilm-40", "soilm-100", "soilm-200"]
    layer_depths = [.1, .3, .6, 1.]

    '''
    sl_hist_files = sorted([
        p for p in sportlis_dir.iterdir()
        if "sportlis_HIST" in p.name
        ])
    sl_pct_files = sorted([
        p for p in sportlis_dir.iterdir()
        if "sportlis_percentile" in p.name
        ])
    sample_file = "sportlis_HIST_201603180000_d01.grb"
    with pygrib.open(sportlis_dir.joinpath(sample_file)) as pgf:
        lat,lon = pgf.message(1).latlons()

    print(lat.shape, lon.shape)

    pir,sub_slice = get_index_raster_from_polygons(
            latitudes=lat,
            longitudes=lon,
            shapefile=shapefile,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            return_subgrid_slices=True,
            )

    plot_geo_ints(
            int_data=pir,
            lat=lat[*sub_slice],
            lon=lon[*sub_slice],
            geo_bounds=None,
            latlon_ticks=False,
            cbar_ticks=False,
            int_labels=None, ## functionally color bar tick labels
            fig_path=None,
            colors=None,
            show=True,
            plot_spec={}
            )
    exit(0)
    '''

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
    ## indices of soil layers wrt the LIS_HIST; add one due to GRIB standard
    hist_soilm_record_idxs = [rlabels_hist.index(l) for l in soilm_labels]
    ## labels of corresponding soil layers in the percentile file
    pct_soilm_record_idxs = list(range(4))

    smvi = get_sportlis_smvi_series(
        hist_file_dir=sportlis_dir,
        percentile_file_dir=sportlis_dir,
        hist_record_indices=hist_soilm_record_idxs,
        percentile_record_indices=pct_soilm_record_idxs,
        layer_depths=layer_depths,
        start_time=start_time,
        end_time=end_time,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        #smvi_config=SMVIConfig(),
        nworkers=4,
        ngroups=10,
        )
    print(f"{smvi.shape=}")
