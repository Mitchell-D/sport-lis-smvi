import numpy as np
import geopandas as gpd
from pathlib import Path
import shapely
from shapely.geometry import Point,Polygon
from shapely.strtree import STRtree
from time import perf_counter

from helpers import get_bounding_latlon_slice

def apply_by_polygon(dataset, poly_int_raster, agg_func,
        dtype=np.float32, poly_oob_value=-1, output_oob_value=np.nan):
    """
    Given a (N,Y,X,F) dataset and a (Y,X) polygon integer raster array
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
                func1d=agg_func, axis=1, arr=dataset[:,m_pix]
                )[:,np.newaxis]
        out[:,m_pix,:] = tmp
    return out

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
        print(f"{perf_counter():.3f} Reading shapefile ")
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
        print(f"{perf_counter():.3f} Initializing STR Tree ")
    ## make an STR tree of the polygons so that it's efficient to rule out
    ## inclusion of pixels that are strictly outside the minimum bounding
    ## rectangle. See linked document:
    ## https://ia600709.us.archive.org/13/items/nasa_techdoc_19970016975/19970016975.pdf
    tree = STRtree(polygons)

    if debug:
        print(f"{perf_counter():.3f} Grouping by polygons ")
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
