"""
MESH implementation for calculating on PPI data.
This algorthim was originally developed by Witt et al. 1998 and modified by Murillo and Homeyer 2019 

Joshua Soderholm - 15 August 2020
"""

import copy
import numpy as np

def add_pyart_metadata(radar, variable_name, metadata_dict, skip_key="data"):
    """
    For each key in metadata_dict, a new attribute is created in sweep_ds with the key value

    Parameters
    ----------
    radar : class
        pyart radar object
    variable_name: string
        name of variable in sweep_ds to update
    metadata_dict: dict
        dictionary containing keys and values to add into sweep_ds
    skip_key: string
        names of key to skip in metadata_dict

    Returns
    -------
    radar : class
        pyart radar object

    """

    for key_name in metadata_dict.keys():
        if key_name != skip_key:
            radar.fields[variable_name][key_name] = metadata_dict[key_name]
    return radar

def thresholded_hail_correction(n, radar_id=None):
    """
    Applies a thresholded correction to hail size estimations
    (supports scalars and NumPy arrays).
    
    Parameters
    ----------
    n : array_like
        Input array of values to correct
    radar_id : str, optional
    """
    n = np.asarray(n)
    # KSGP picking up more reflectivity
    radar_diff = 23 if radar_id == 'KGSP' else 19 
    corrected = np.where(n < 40, np.maximum(n - radar_diff, 0), n)
    mask = radar_diff * (1 - (n - 40) / 30)
    corrected = np.where((n >= 40) & (n < 70), n - mask, corrected)

    return corrected

def safe_log(x, eps=1e-10):
    """
    Safe log function

    Parameters
    ----------
    x: numpy array

    Returns
    -------
    result : numpy array
"""

    result = np.where(x > eps, x, -10)
    np.log(result, out=result, where=result > 0)
    return result


def main(
    radar,
    reflectivity_fname,
    temp_levels,
    ke_fname="ke",
    shi_fname="shi",
    mesh_fname="mesh",
    posh_fname="posh",
    radar_band="S",
    min_range=10,
    max_range=150,
    mesh_method="mh2019_75",
    correct_cband_refl=True,
    minimum_sweeps_raise_expection=4,
    minimum_sweeps_raise_warning=8,
    column_shift_maximum=2500,
    radar_id=None,
):
    """
    PyART Wrapper for PPI MESH

    Parameters:
    ===========
    radar: class
        pyart radar object
    reflectivity_fname: string
        name of reflectivity field
    temp_levels : list of length 2
        height above sea level (m) of the freezing level and -20C level (in any order)
    ke_fname: string
        name of ke field
    shi_fname: string
        name of shi field
    mesh_fname: string
        name of mesh field
    posh_fname: string
        name of posh field
    radar_band: str
        radar frequency band (either C or S)
    min_range: int
        minimum surface range for MESH retrieval (m)
    max_range: int
        maximum surface range for MESH retrieval (m)
    mesh_method : string
        either witt1998, mh2019_75 or mh2019_95. see more information below
    correct_cband_refl: logical
        flag to trigger C band hail reflectivity correction (if radar_band is C)
    minimum_sweeps_raise_expection: int
        minimum number of sweeps to raise an exception
    minimum_sweeps_raise_warning: int
        minimum number of sweeps to raise a warning
    column_shift_maximum: float
        maximum horizontal distance a column can shift by
    radar_id: str
        radar station identifier
    Returns:
    ========
    mesh_points: dict
        Dictionary containing 'lats', 'lons', and 'mesh_values' arrays
    """

    # init radar fields
    empty_radar_field = {
        "data": np.zeros((radar.nrays, radar.ngates)),
        "units": "",
        "long_name": "",
        "description": "",
        "comments": "",
    }
    radar.add_field(ke_fname, copy.deepcopy(empty_radar_field))
    radar.add_field(shi_fname, copy.deepcopy(empty_radar_field))
    radar.add_field(mesh_fname, copy.deepcopy(empty_radar_field))
    radar.add_field(posh_fname, copy.deepcopy(empty_radar_field))
    # build datasets
    reflectivity_dataset = []
    elevation_dataset = []
    azimuth_dataset = []
    range_dataset = []
    elevation_dataset = []
    radar_altitude = radar.altitude["data"][0]
    for sweep_idx in range(radar.nsweeps):
        reflectivity_dataset.append(
            radar.get_field(sweep_idx, reflectivity_fname, copy=True).filled(np.nan)
        )
        azimuth_dataset.append(radar.get_azimuth(sweep_idx))
        range_dataset.append(radar.range["data"])
        elevation_dataset.append(radar.fixed_angle["data"][sweep_idx])

    # run retrieval
    mesh_dict = ppi_calc(
        reflectivity_dataset,
        elevation_dataset,
        azimuth_dataset,
        range_dataset,
        radar_altitude,
        temp_levels,
        radar_band=radar_band,
        min_range=min_range,
        max_range=max_range,
        mesh_method=mesh_method,
        correct_cband_refl=correct_cband_refl,
        minimum_sweeps_raise_expection=minimum_sweeps_raise_expection,
        minimum_sweeps_raise_warning=minimum_sweeps_raise_warning,
        column_shift_maximum=column_shift_maximum,
        radar_id=radar_id,
    )
    
    # Get the lowest sweep index where MESH is valid
    sweep0_idx = np.argmin(elevation_dataset)
    
    # Convert from radar polar coordinates to lat/lon using pyart's geographic transform
    lons, lats, _ = radar.get_gate_lat_lon_alt(sweep0_idx)
    
    # Get MESH values and mask
    mesh_values = mesh_dict['data']
    valid_mask = ~np.isnan(mesh_values)
    
    # Create output dictionary with only valid points
    mesh_points = [
        {
            'lat': float(lat),
            'lon': float(lon),
            'mesh_value': float(mesh_value)
        }
        for lat, lon, mesh_value in zip(lats[valid_mask], lons[valid_mask], mesh_values[valid_mask])
    ]

    return mesh_points


def _antenna_to_arc(ranges, elevation):
    """
    Return the great circle distance directly below the radar beam and the
    altitude of the radar beam.
    ----------
    ranges : 1d array
        Distances to the center of the radar gates (bins) in meters.
    elevation : float
        Elevation angle of the radar in degrees.
    Returns
    -------
    s: 1d array
        Distance along the great circle for each radar bin (units: meters)
    z: 1d array
        Altitude above radar level for each radar bin (units: meters)
    Notes
    -----
    The calculation for Cartesian coordinate is adapted from equations
    2.28(b) and 2.28(c) of Doviak and Zrnic [1]_ assuming a
    standard atmosphere (4/3 Earth's radius model).
    .. math::
        z = \\sqrt{r^2+R^2+2*r*R*sin(\\theta_e)} - R
        s = R * arcsin(\\frac{r*cos(\\theta_e)}{R+z})
    Where r is the distance from the radar to the center of the gate,
    :math:`\\theta_a` is the azimuth angle, :math:`\\theta_e` is the
    elevation angle, s is the arc length, and R is the effective radius
    of the earth, taken to be 4/3 the mean radius of earth (6371 km).
    References
    ----------
    .. [1] Doviak and Zrnic, Doppler Radar and Weather Observations, Second
        Edition, 1993, p. 21.
    """
    theta_e = elevation * np.pi / 180.0  # elevation angle in radians.
    R = 6371.0 * 1000.0 * 4.0 / 3.0  # effective radius of earth in meters.
    r = ranges

    z = (r**2 + R**2 + 2.0 * r * R * np.sin(theta_e)) ** 0.5 - R
    s = R * np.arcsin(r * np.cos(theta_e) / (R + z))  # arc length in m.
    return s, z


def _calc_dz(column_z):
    """
    Calculate altitude difference between elements in a 1d array
    Takes into account the boundaries

    Parameters
    ----------
    column_z : 1darray
        altitude of column samples

    Returns
    -------
    dz : 1darray
        Difference between altitude elements
    """
    # need at least two values
    dz = []
    n_ppi = len(column_z)
    for i in range(n_ppi):
        # calculate dz for use in shi calc
        if i == 0:
            value = column_z[i + 1] - column_z[i]
        elif (i != 0) & (i != n_ppi - 1):
            value = (column_z[i + 1] - column_z[i - 1]) / 2
        else:
            value = column_z[i] - column_z[i - 1]
        dz.append(value)
    dz = np.array(dz)
    return dz


def ppi_calc(
    reflectivity,
    elevation,
    azimuth,
    rangebin,
    radar_altitude,
    levels,
    radar_band="S",
    min_range=10,
    max_range=150,
    mesh_method="mh2019_75",
    correct_cband_refl=True,
    minimum_sweeps_raise_expection=4,
    minimum_sweeps_raise_warning=8,
    column_shift_maximum=2500,
    radar_id=None,
):
    """
    Adapted from Witt et al. 1998 and Murillo and Homeyer 2019

    Parameters
    ----------
    reflectivity : list of 2D ndarrays
        list where each element is the sweep reflectivity data in an array with dimensions (azimuth, range)
    elevation: 1d ndarray of floats
        ndarray where each element is the fixed elevation angle of the sweep
    azimuth: list of 1D ndarrays
        list where each element is the sweep azimuth angles
    rangebin: list of 1D ndarrays
        list where each element is the sweep range distances
    radar_altitude: float
        altitude of radar AMSL
    levels : list of length 2
        height above sea level (m) of the freezing level and -20C level (in any order)
    radar_band: str
        radar frequency band (either C or S)
    min_range: int
        minimum surface range for MESH retrieval (m)
    max_range: int
        maximum surface range for MESH retrieval (m)
    mesh_method : string
        either witt1998, mh2019_75 or mh2019_95. see more information below
    correct_cband_refl: logical
        flag to trigger C band hail reflectivity correction (if radar_band is C)
    minimum_sweeps_raise_expection: int
        minimum number of sweeps to raise an exception
    minimum_sweeps_raise_warning: int
        minimum number of sweeps to raise a warning
    column_shift_maximum: float
        maximum horizontal distance a column can shift by
    radar_id: str
        radar station identifier
    Returns
    -------
    output_fields : dictionary
        Dictionary of output fields (KE, SHI, MESH, POSH)
    """

    # require C or S band
    if radar_band not in ["C", "S"]:
        raise ValueError("radar_band must be a string of value C or S")
    # require levels
    if levels is None:
        raise ValueError("Missing levels data for freezing level and -20C level")

    # Rain/Hail dBZ boundaries
    z_l = 40
    z_u = 50

    # This dummy proofs the user input. The melting level will always
    # be lower in elevation than the negative 20 deg C isotherm
    meltlayer = np.min(levels)
    neg20layer = np.max(levels)

    # sort by fixed angle
    sort_idx = list(np.argsort(elevation))
    reflectivity_dataset = [reflectivity[i] for i in sort_idx]
    elevation_dataset = [elevation[i] for i in sort_idx]
    azimuth_dataset = [azimuth[i] for i in sort_idx]
    range_dataset = [rangebin[i] for i in sort_idx]

    # require more than one sweep
    if len(elevation_dataset) <= minimum_sweeps_raise_expection:
        print(f"Warning: Skipping scan - requires more than {minimum_sweeps_raise_expection} sweeps to calculate MESH")
        return None
    elif len(elevation_dataset) < minimum_sweeps_raise_warning:
        print(f"Warning: Number of sweeps ({len(elevation_dataset)}) is less than {minimum_sweeps_raise_warning}")

    # sweep must be sorted from lowest to highest elevation
    dx = np.diff(elevation_dataset)
    if np.all(dx <= 0):
        print("Warning: Skipping scan - sweeps are not sorted in increasing monotonic order")
        return None

    # Initialize sweep coords
    sweep0_nbins = len(range_dataset[0])
    n_ppi = len(elevation_dataset)
    z_dataset = (
        []
    )  # list (dim: elevation) of 1d array (dim: range) for each sweep, altitude above ground level (m) of each range bin
    s_dataset = (
        []
    )  # list (dim: elevation) of 1d array (dim: range) for each sweep, great circle arc distance (m) of each radar bin
    wt_dataset = (
        []
    )  # list (dim: elevation) of 1d array (dim: range) for each sweep, temperature weighting value
    hail_ke_dataset = (
        []
    )  # list (dim: elevation) of 2d array (dim: azimuth, range) for each sweep, hail kinetic energy
    hail_refl_correction_description = ""
    for i in range(n_ppi):
        # calculate cartesian coordinates
        s, z = _antenna_to_arc(range_dataset[i], elevation_dataset[i])
        s_dataset.append(s)
        z_dataset.append(z + radar_altitude)
        # calc temperature based weighting function
        wt = (z_dataset[i] - meltlayer) / (neg20layer - meltlayer)
        wt[z_dataset[i] <= meltlayer] = 0
        wt[z_dataset[i] >= neg20layer] = 1
        wt[wt < 0] = 0
        wt[wt > 1] = 1
        wt_dataset.append(wt)
        # apply C band correction
        if radar_band == "C" and correct_cband_refl:
            reflectivity_dataset[i] = reflectivity_dataset[i] * 1.113 - 3.929
            hail_refl_correction_description = (
                "C band hail reflectivity correction applied"
                " from Brook et al. 2023 https://arxiv.org/abs/2306.12016"
            )
        # calc weights for hail kenetic energy
        reflectivity_weights = (reflectivity_dataset[i] - z_l) / (z_u - z_l)
        reflectivity_weights[reflectivity_dataset[i] <= z_l] = 0
        reflectivity_weights[reflectivity_dataset[i] >= z_u] = 1
        reflectivity_weights[reflectivity_weights < 0] = 0
        reflectivity_weights[reflectivity_weights > 1] = 1
        # limit on DBZ
        reflectivity_dataset[i][reflectivity_dataset[i] > 100] = 100
        reflectivity_dataset[i][reflectivity_dataset[i] < -100] = -100
        # calc hail kenetic energy
        hail_ke = (
            (5.0e-6) * 10 ** (0.084 * reflectivity_dataset[i]) * reflectivity_weights
        )
        hail_ke[np.isnan(hail_ke)] = 0
        hail_ke_dataset.append(hail_ke)

    # generate arc range and dz lookup (note these have different dimensions to the dimension variables)
    dz_dataset = (
        []
    )  # list (dim: range) where each element represents a range bin, 1d array (dim: elevation) where each element represents a sweep, altitude dz for shi integration (m)
    s_lookup_dataset = (
        []
    )  # list (dim: range) where each element represents an the range bin index to use from each sweep above sweep0. ASSUMES ORDERS SWEEP ELEVATION
    for rg_idx in range(sweep0_nbins):
        s_lookup = [0]
        column_z = [z_dataset[0][rg_idx]]
        for sweep_idx in range(1, n_ppi, 1):
            dist_array = np.abs(s_dataset[0][rg_idx] - s_dataset[sweep_idx])
            closest_rng_idx = np.argmin(dist_array)
            # skip sweeps where the horizontal shift is greater than column_shift_maximum (removes birdbaths and when base scan max range is greater than all other scans)
            if dist_array[closest_rng_idx] < column_shift_maximum:
                s_lookup.append(closest_rng_idx)
                column_z.append(z_dataset[sweep_idx][closest_rng_idx])
            # else:
            #     print('skipping', 'distance check', dist_array[closest_rng_idx], 'range idx', rg_idx, 'sweep idx', sweep_idx)
        # check if at least two valid values in the column exists
        if len(s_lookup) > 1:
            s_lookup_dataset.append(np.array(s_lookup))
            dz_dataset.append(_calc_dz(column_z))
        else:
            s_lookup_dataset.append(None)
            dz_dataset.append(None)

    # calculate shi on lowest sweep coordinates
    shi = np.zeros((len(azimuth_dataset[0]), len(range_dataset[0])))
    shi_mask = np.zeros((len(azimuth_dataset[0]), len(range_dataset[0])), dtype=bool)
    
    # Vectorize range mask calculation
    range_mask = (s_dataset[0] < min_range * 1000) | (s_dataset[0] > max_range * 1000)
    shi_mask[:, range_mask] = True
    
    # Vectorize column validity mask
    valid_columns = np.array([lookup is not None for lookup in s_lookup_dataset])
    shi_mask[:, ~valid_columns] = True
    
    # Pre-calculate all azimuth differences and closest indices
    closest_az_indices = {}
    for sweep_idx in range(1, n_ppi):
        az_diff = np.abs(azimuth_dataset[sweep_idx][:, np.newaxis] - azimuth_dataset[0])
        closest_az_indices[sweep_idx] = np.argmin(az_diff, axis=0)
    
    # Process all valid columns at once
    valid_az_indices, valid_rg_indices = np.where(~shi_mask)
    
    for sweep_idx in range(n_ppi):
        if sweep_idx == 0:
            # Process first sweep directly
            column_elements = (
                hail_ke_dataset[0][valid_az_indices, valid_rg_indices] * 
                wt_dataset[0][valid_rg_indices]
            )
            dz_values = np.array([dz_dataset[rg][0] for rg in valid_rg_indices])
            shi_elements = column_elements * dz_values
            
        else:
            # Get closest azimuth indices for this sweep
            sweep_az_indices = closest_az_indices[sweep_idx][valid_az_indices]
            
            # Get range indices for this sweep
            sweep_rg_indices = np.array([
                s_lookup_dataset[rg][sweep_idx] if s_lookup_dataset[rg] is not None else -1
                for rg in valid_rg_indices
            ])
            
            # Create valid mask for this sweep
            valid_sweep_mask = (
                (sweep_rg_indices != -1) & 
                (np.abs(azimuth_dataset[sweep_idx][sweep_az_indices] - 
                       azimuth_dataset[0][valid_az_indices]) <= 1)
            )
            
            if not np.any(valid_sweep_mask):
                continue
                
            # Calculate elements for valid points
            valid_points = np.where(valid_sweep_mask)[0]
            sweep_elements = np.zeros(len(valid_az_indices))
            
            sweep_elements[valid_points] = (
                hail_ke_dataset[sweep_idx][sweep_az_indices[valid_points], 
                                         sweep_rg_indices[valid_points]] *
                wt_dataset[sweep_idx][sweep_rg_indices[valid_points]]
            )
            
            # Get dz values for this sweep
            dz_values = np.array([
                dz_dataset[rg][sweep_idx] if dz_dataset[rg] is not None else 0
                for rg in valid_rg_indices
            ])
            
            shi_elements = sweep_elements * dz_values
            
        # Add to SHI grid
        shi[valid_az_indices, valid_rg_indices] += 0.1 * shi_elements

    # calc maximum estimated severe hail (mm)
    if (
        mesh_method == "witt1998"
    ):  # 75th percentil fit from witt et al. 1998 (fitted to 147 reports)
        mesh = 2.54 * shi**0.5
        mesh_description = "Maximum Estimated Size of Hail retreival developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 "
        mesh_comment = (
            "75th percentile fit using 147 hail reports; only valid in the first sweep"
        )

    elif (
        mesh_method == "mh2019_75"
    ):  # 75th percentile fit from Muillo and Homeyer 2019 (fitted to 5897 reports)
        mesh = thresholded_hail_correction(15.096 * shi**0.206, radar_id)
        mesh_description = "Maximum Estimated Size of Hail retreival originally developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 and recalibrated by Murillo and Homeyer (2021) doi:10.1175/JAMC-D-20-0271.1 "
        mesh_comment = (
            "75th percentile fit using 5897 hail reports; only valid in the first sweep"
        )
    elif (
        mesh_method == "mh2019_95"
    ):  # 95th percentile fit from Muillo and Homeyer 2019 (fitted to 5897 reports)
        mesh = 22.157 * shi**0.212
        mesh_description = "Maximum Estimated Size of Hail retreival originally developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 and recalibrated by Murillo and Homeyer (2021) doi:10.1175/JAMC-D-20-0271.1 "
        mesh_comment = (
            "95th percentile fit using 5897 hail reports; only valid in the first sweep"
        )
    else:
        raise ValueError(
            "unknown MESH method selects, please use witt1998, mh2019_75 or mh2019_95"
        )

    # calc warning threshold (J/m/s) NOTE: freezing height must be in km
    warning_threshold = 57.5 * (meltlayer / 1000) - 121

    # calc probability of severe hail (POSH) (%)
    posh = 29 * safe_log(shi / warning_threshold) + 50
    posh = np.real(posh)
    posh[posh < 0] = 0
    posh[posh > 100] = 100

    # mask outside of coverage with nan
    posh[shi_mask] = np.nan
    shi[shi_mask] = np.nan
    mesh[shi_mask] = np.nan

    # add grids to radar object
    # unpack E into cfradial representation

    mesh_dict = {
        "data": mesh,
        "units": "mm",
        "long_name": "Maximum Expected Size of Hail using " + mesh_method,
        "description": mesh_description + hail_refl_correction_description,
        "comments": mesh_comment,
    }

    # return output_fields dictionary
    return mesh_dict