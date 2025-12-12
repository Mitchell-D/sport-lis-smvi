""" Sanity check for visualizing poly raster pickle files """
import numpy as np
import pickle as pkl
import geopandas as gpd
from pathlib import Path
from plotting import plot_geo_ints

locales = ["al", "ca", "dakotas", "fl", "in", "la", "matl", "midwest", "mt",
        "nc", "neus", "nwus", "swus", "conus"]
#poly_raster_dir = Path("/usr/people/mdodson/sport-lis-smvi/data/poly2")
#fig_dir = Path("/usr/people/mdodson/sport-lis-smvi/figures/poly2")
poly_raster_dir = Path("/usr/people/mdodson/sport-lis-smvi/data/poly")
fig_dir = Path("/usr/people/mdodson/sport-lis-smvi/figures/poly")
shapefile_path = Path("data/shapefiles/c_15au13.shp")

for l in locales:
    poly_raster_path = poly_raster_dir.joinpath(
            f"poly-raster_counties_{l}.pkl")
    if not poly_raster_path.exists():
        print(f"Skipping not found: {poly_raster_path.as_posix()}")
        continue
    pir,metadata,sub_slice,(lat,lon) = pkl.load(poly_raster_path.open("rb"))
    gdf = gpd.read_file(shapefile_path)
    polys = [gdf["geometry"][md["poly_idx"]] for md in metadata]
    fig_path = fig_dir.joinpath(f"poly-raster_counties_{l}.png")
    print(lat.shape, lon.shape, pir.shape)
    try:
        plot_geo_ints(
            int_data=np.where(pir==-1,np.nan,pir),
            lat=lat,
            lon=lon,
            int_labels=None,
            shapes=polys,
            cbar_ticks=False,
            latlon_ticks=False,
            plot_spec={
                "title":f"county polygons in {l} domain",
                "cbar_pad":.05,
                "cbar_shrink":.8,
                "cbar_orient":"horizontal",
                "cmap":"prism",
                "shape_params":{
                    "edgecolor":"silver",
                    "facecolor":"none",
                    "alpha":.85,
                    },
                "interpolation":"none",
                },
            show=False,
            fig_path=fig_path,
            )
        print(f"Generated {fig_path.as_posix()}")
    except Exception as e:
        print(f"Failed ({l}): {e}")
