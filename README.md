# sport-lis-smvi

Modules for calculating and visualizing soil moisture volatility
index based on SPoRT-LIS data
<p align="center">
   <img src="https://raw.githubusercontent.com/Mitchell-D/sport-lis-smvi/main/figures/sportlis_smvi_la_20230606-20231031_anim_mitchell-binary.gif" width=60%>
   <img src="https://raw.githubusercontent.com/Mitchell-D/sport-lis-smvi/main/figures/sportlis_smvi_la_20230606-20231031_anim_mitchell-pixelwise.gif" width=60%>
</p>

## adpating this code

There are two options for executing the same algorithm and generating
essentially the same plots. In both cases, start by ensuring you have
a python environment that contains the dependencies listed in
env.yml.

### option 1

1. Copy main\_smvi.py, get\_poly\_raster.py, get\_sportlis\_smvi.py,
   plotting.py, and helpers.py into the working directory.

2. Verify the existence of and note paths to a shapefile for grouping
   pixels, collections of SPoRT-LIS percentile and HIST files, and a
   numpy file containing the SPoRT-LIS latlon domain (available in
   this repository at data/sportlis\_latlon.py).

3. Create a new empty directory where polygon raster pickle files
   will be stored for each region on the first pass for future use.
   To save time on SC1, recursively copy the following directory of
   raster pkls: /usr/people/mdodson/sport-lis-smvi/data/poly

5. In main\_smvi.py, modify the shebang on the first line to point to
   your environment binary. Then, under `if __name__=="__main__":`,
   modify the paths and naming templates to point to the files and
   directories from steps 2 and 3.

6. Below the paths, assess the options for plotting, integrating
   layers, load-balancing with `nworkers` and `ngroups`, etc, and
   make changes if desired. You may also wish to add lat/lon bounding
   boxes for additional regional zooms in the `geog` dictionary in
   global scope at the top of main\_smvi.py

7. Change the permissions of main\_smvi.py to enable execution.

8. Read the CLI options with `main_smvi.py -h`, then use the CLI to
   calculate SMVI and generate figures.

### option 2

1. From the jon directory of the repo, copy lis\_utils.py and
   sportlis\_smvi.py into your working directory.

2. Verify the existence of and note paths to a shapefile for grouping
   pixels, and collections of SPoRT-LIS percentile and HIST files.

3. Create a new empty directory where polygon raster pickle files
   will be stored for each region on the first pass for future use.
   To save time on SC1, recursively copy the following directory
   of raster pkls: /usr/people/mdodson/sport-lis-smvi/poly-jon

5. In lis\_utils.py, modify `POLY_RASTER_DIR` to point to the empty
   directory created in step 3, and modify `county_shp` to point to
   the shapefile. Optionally modify the `geog` dict to include
   additional geographic bounding boxes for analysis.

6. In sportlis\_smvi.py point the `DATADIR`, `GRIDDIR`, and `OUTDIR`
   directories to the HIST, percentile, and figure output
   directories, respectively. Then define a `region_list` containing
   string keys of the `lis_utils.geog` dict indicating which subgrids
   will be analyzed.

7. Execute sportlis\_smvi.py with your python environment, providing
   a single YYYYMMDD positional argument indicating which day to
   calculate SMVI for. Example: `python sportlis_smvi.py 20230901`
