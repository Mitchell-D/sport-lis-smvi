import numpy as np
import pickle as pkl
import geopandas as gpd
from pathlib import Path

from get_poly_raster import get_poly_raster,apply_by_polygon
from plotting import plot_geo_ints

if __name__=="__main__":
    data_dir = Path("data")
    ## counties shapefile
    shapefile = data_dir.joinpath("shapefiles/c_15au13.shp")
    ## retrieve the full-sized SPoRT-LIS latitude and longitude arrays
    latlon = np.load(data_dir.joinpath("sportlis_latlon.npy"))
    ## (T,Y,X,L) numpy int array for T times, Y/X lat/lon, L levels
    ## OOB values are -1, valid non-SMVI values are 0, SMVI values are 1
    smvi = np.load(data_dir.joinpath("smvi_demo_sample.npy"))
    m_valid = (smvi != -1)

    activation_threshold = .3

    pir,metadata,sub_slice = get_poly_raster(
            latitudes=latlon[...,0],
            longitudes=latlon[...,1],
            shapefile=shapefile,
            lat_bounds=(28,34),
            lon_bounds=(-96,-87.8),
            shapefile_columns=[
                "STATE","CWA","COUNTYNAME","FIPS","TIME_ZONE",
                "FE_AREA","LON","LAT","Shape_Area"],
            return_subgrid_slices=True,
            debug=False,
            )

    ## apply a function to each polygon which returns 1 when the fraction of
    ## pixels in the polygon is greater than activation_threshold, 0 otherwise.
    poly_smvi = apply_by_polygon(
        dataset=smvi,
        poly_int_raster=pir,
        agg_func=lambda x:(np.average(x)>activation_threshold).astype(int),
        output_oob_value=np.nan,
        )
    ## modify the integers so that 0 is out of bounds, 1 is in-bounds but SMVI
    ## below threshold, and 2 represents polygons exceeding the threshold
    poly_smvi = np.where(m_valid, poly_smvi+1, 0)

    ## get the relevant counties from the indeces in the metadata
    gdf = gpd.read_file(shapefile)
    polys = [gdf["geometry"][md["poly_idx"]] for md in metadata]
    ## extract random timestep at 0-100cm level as an example
    smvi_sample = poly_smvi[3,...,2]
    ## plot the example
    plot_geo_ints(
        int_data=smvi_sample,
        lat=latlon[...,0][*sub_slice],
        lon=latlon[...,1][*sub_slice],
        int_labels=[
            "Out of Domain",
            f"SMVI Fraction <= {activation_threshold}",
            f"SMVI Fraction > {activation_threshold}",
            ],
        fig_path=Path("smvi_demo.png"),
        latlon_ticks=False,
        shapes=polys,
        cbar_ticks=True,
        plot_spec={
            "cbar_pad":0.02,
            "cbar_orient":"horizontal",
            "cbar_shrink":.8,
            "cbar_fontsize":14,
            "tick_frequency":12,
            "tick_rotation":45,
            "title":f"Counties with >{activation_threshold*100}% " + \
                    "SMVI (0-100cm)",
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
