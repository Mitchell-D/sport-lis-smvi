import numpy as np
import pygrib
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from dataclasses import dataclass
from multiprocessing import Pool
from datetime import datetime,timedelta

from helpers import get_bounding_latlon_slice

@dataclass
class SMVIConfig:
    """
    Configuration object defining a set of SMVI rules.

    SMVI is a boolean function such that it returns True when:

     - Daily big window average is greater than small window average for a
       number of days (drying_days).
     - The climatological percentile on the last day is below a cutoff
    """
    big_window_size:int=20
    small_window_size:int=5
    drying_days:int=20
    last_day_percentile_cutoff:float=20

def get_sportlis_smvi(
        hist_file_dir:Path, percentile_file_dir:Path,
        hist_record_indices:tuple, percentile_record_indices:tuple,
        start_time:datetime, end_time:datetime,
        lat_bounds:tuple, lon_bounds:tuple,
        smvi_config:SMVIConfig=None,
        integrate_layers=True, layer_depths=None,
        nworkers=1, ngroups=4,
        latitudes=None, longitudes=None,
        hist_file_pattern="sportlis_HIST_{yyyymmdd}0000_d01.grb",
        percentile_file_pattern="sportlis_vsm_percentile_{yyyymmdd}.grb2",
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

    :@return: 2-tuple (smvi, dates) where smvi is an array of SMVI values
        shaped like (N,Y,X,F) for N days between start_time and end_time
        inclusively, and with F soil layers. dates is a list of datetime
        objects corresponding to each of the N data values
    """
    fphist = hist_file_pattern
    fppct = percentile_file_pattern

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
    pct_datetimes = []
    lag_days = smvi_config.big_window_size + smvi_config.drying_days - 2
    t = start_time - timedelta(days=lag_days)
    while t <= end_time:
        tstr = t.strftime("%Y%m%d")
        hist_dates.append(tstr)
        if t >= start_time:
            pct_dates.append(tstr)
            pct_datetimes.append(t)
        t += timedelta(days=1)

    ## explicitly calculate the needed file names given the earthdata scheme,
    ## and make sure they exist in the provided directories.
    req_hist = [hist_file_dir.joinpath(fphist.format(yyyymmdd=tstr))
                for tstr in hist_dates]
    req_pct = [percentile_file_dir.joinpath(fppct.format(yyyymmdd=tstr))
                for tstr in pct_dates]
    for p in req_hist + req_pct:
        if not p.exists():
            raise ValueError(f"Missing required file: {p.as_posix()}")

    ## assume the grids are all uniform and extract coordinate arrays
    griblat,griblon = None,None
    if latitudes is None or longitudes is None:
        with pygrib.open(req_hist[0]) as pgf:
            griblat,griblon = pgf.message(1).latlons()
    lat = griblat if latitudes is None else latitudes
    lon = griblon if longitudes is None else longitudes

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
    return np.concatenate(smvi, axis=0),pct_datetimes

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
                ## nldas convention is to index latitude low to high
                x = pgf.readline().values.data[*slice_bounds][::-1]
                ## hist files use 9999. as a mask value
                if m_valid is None:
                    m_valid = ~np.isclose(x, 9999)
                tmp_feats.append(x[m_valid])
            hist.append(np.stack(tmp_feats, axis=-1))
    hist = np.stack(hist, axis=0)

    ## read all the percentile data into memory
    pct = []
    for p in pct_paths:
        with pygrib.open(p) as pgf:
            tmp_feats = []
            for rix in pct_rixs:
                pgf.seek(rix)
                ## nldas convention is to index latitude low to high
                ## but i don't like that
                x = pgf.readline().values.data[*slice_bounds][::-1]
                tmp_feats.append(x[m_valid])
            pct.append(np.stack(tmp_feats, axis=-1))
    pct = np.stack(pct, axis=0)

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
    smvi = np.full(
            (m_pct.shape[0], *m_valid.shape, m_pct.shape[-1]), -1, dtype=int)
    smvi[:,m_valid,:] = (m_pct & m_drying).astype(int)
    return smvi
