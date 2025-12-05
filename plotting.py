import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from datetime import datetime,timedelta
from cartopy.mpl.ticker import LatitudeFormatter,LongitudeFormatter
from matplotlib.colors import ListedColormap

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
            pad=ps.get("cbar_pad", 0.05),
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

def plot_geo_scalar(data, latitude, longitude, bounds=None, plot_spec={},
             latlon_ticks=False, show=False, fig_path=None,
             use_contours=False):
    """
    Plot a gridded scalar value on a geodetic domain, using cartopy for borders
    """
    ps = {"xlabel":"", "ylabel":"", "marker_size":4,
          "cmap":"jet_r", "text_size":12, "title":"",
          "norm":"linear","figsize":(12,12), "marker":"o", "cbar_shrink":1.,
          "map_linewidth":2}
    plt.clf()
    ps.update(plot_spec)
    plt.rcParams.update({"font.size":ps["text_size"]})

    ax = plt.axes(projection=ccrs.PlateCarree())
    fig = plt.gcf()
    if bounds is None:
        bounds = [np.amin(longitude), np.amax(longitude),
                  np.amin(latitude), np.amax(latitude)]
    ax.set_extent(bounds, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, linewidth=ps.get("map_linewidth"))
    #ax.add_feature(cfeature.LAKES, linewidth=ps.get("map_linewidth"))
    #ax.add_feature(cfeature.RIVERS, linewidth=ps.get("map_linewidth"))

    ax.set_title(ps.get("title"), fontsize=ps.get("fontsize_title", 18))
    ax.set_xlabel(ps.get("xlabel"), fontsize=ps.get("fontsize_labels", 14))
    ax.set_ylabel(ps.get("ylabel"), fontsize=ps.get("fontsize_labels", 14))

    if use_contours:
        scat = ax.contourf(
                longitude,
                latitude,
                data,
                cmap=ps.get("cmap"),
                norm=ps.get("norm"),
                vmin=ps.get("vmin"),
                vmax=ps.get("vmax"),
                )
    else:
        scat = ax.pcolormesh(
                longitude,
                latitude,
                data,
                cmap=ps.get("cmap"),
                norm=ps.get("norm"),
                vmin=ps.get("vmin"),
                vmax=ps.get("vmax"),
                )

    if latlon_ticks:
        lonmin,lonmax,latmin,latmax = bounds
        frq = ps.get("tick_frequency", 1)
        ax.set_yticks(np.linspace(latmin,latmax,data.shape[0])[::frq],
                crs=ccrs.PlateCarree())
        ax.set_xticks(np.linspace(lonmin,lonmax,data.shape[1])[::frq],
                crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(rotation=ps.get("tick_rotation", 0))

    ax.add_feature(cfeature.BORDERS, linewidth=ps.get("map_linewidth"),
                   zorder=120)
    ax.add_feature(cfeature.STATES, linewidth=ps.get("map_linewidth"),
                   zorder=120)
    ax.coastlines()
    fig.colorbar(
            scat,
            ax=ax,
            shrink=ps.get("cbar_shrink"),
            label=ps.get("cbar_label"),
            orientation=ps.get("cbar_orient", "vertical"),
            pad=ps.get("cbar_pad", 0.0),
            norm=ps.get("norm"),
            )
    scat.figure.axes[0].tick_params(
            axis="both", labelsize=ps.get("fontsize_labels",14))

    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)
    if show:
        plt.show()
    plt.close()
