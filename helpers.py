import numpy as np

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
