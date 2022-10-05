import base64
from io import BytesIO

from flask import Flask, render_template, request
from matplotlib.figure import Figure
import matplotlib
import matplotlib.ticker as mticker
import datetime
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER, LatitudeLocator
import pandas as pd
import namelist
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.collections import LineCollection

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', img_data='')

def plot_tracks(lon_tc, lat_tc, basin_tc, tc_names, vmax_tc,
                lon_syn_tc, lat_syn_tc, basin_syn_tc, vmax_syn_tc,title_name):
    matplotlib.rcParams.update({'font.size': 15})
    colors = np.asarray([(77, 166, 176), (82, 184, 81), (245, 213, 56),
              (245, 176, 56), (245, 135, 56), (196, 87, 57), (171, 3, 3)]) / 255
    wnd_cmap = LinearSegmentedColormap.from_list('wnd', colors, N=7)
    norm = BoundaryNorm([0, 33, 65, 84, 96, 114, 135, 180], wnd_cmap.N)

    lon_min = 0; lon_max = 359.99
    lat_min = -60.1; lat_max = 60.1
    dlon_label = 30
    lon_lmin = np.floor(lon_min / dlon_label) * dlon_label
    lon_lmax = np.ceil(lon_max / dlon_label) * dlon_label
    xlocs = np.arange(lon_lmin, lon_lmax + dlon_label, dlon_label)
    xlocs_shift = np.copy(xlocs)
    xlocs_shift[xlocs > 180] -= 360

    fig = Figure(figsize=(21,15))
    ax = fig.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    proj = ccrs.PlateCarree(central_longitude=180)
    ax = fig.add_subplot(211, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift,
                      color='gray', alpha=0.3)
    gl.bottom_labels = True
    gl.xlabels_top = False
    gl.left_labels = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    #lon_tc[lon_tc < 0] = lon_tc[lon_tc_b < 0] + 360
    lon_tc_b = np.copy(lon_tc)
    lon_tc_b += 180
    lon_tc_b[lon_tc_b >= 180] = lon_tc_b[lon_tc_b >= 180] - 360
    for k in range(len(tc_names)):
        points = np.array([lon_tc_b[k, :], lat_tc[k, :]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=wnd_cmap, norm=norm)
        lc.set_array(vmax_tc[k, :])
        lc.set_linewidth(5)
        line = ax.add_collection(lc)
        ax.text(lon_tc_b[k, 0], lat_tc[k, 0] - 6, tc_names[k],
                {'size': 13, 'ha': 'left', 'backgroundcolor': [0.5, 0.5, 0.5, 0.8]})
    ax.set_title('Historical Tropical Cyclones: '+title_name)

    # Plot the six basins
    basins = ['NI', 'WP', 'EP', 'SI', 'SP', 'NA']
    basins_long = ['North Indian', 'Western Pacific', 'Eastern Pacific',
                   'South Indian', 'South Pacific', 'North Atlantic']
    for j in range(len(basins)):
        proj = ccrs.PlateCarree(central_longitude=0)
        ax = fig.add_subplot(4, 3, j+7, projection=proj)
        ax.coastlines(resolution='50m')
        ax.set_title(basins_long[j])
        b = TC_Basin(basins[j])
        lon_min, lat_min, lon_max, lat_max =  b.get_bounds()
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                     color='gray', alpha=0.3)
        gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift,
                          color='gray', alpha=0.3)
        gl.xlabels_bottom = True
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Plot tracks for each individual basin.
        b_mask = basin_tc == basins[j]
        for k in range(np.sum(b_mask)):
            points = np.array([lon_tc[b_mask, :][k, :], lat_tc[b_mask, :][k, :]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=wnd_cmap, norm=norm)
            lc.set_array(vmax_tc[b_mask, :][k, :])
            lc.set_linewidth(5)
            line = ax.add_collection(lc)

        # Plot synthetic tracks for each individual basin.
        b_idxs = np.argwhere(basin_syn_tc == basins[j]).flatten()
        if len(b_idxs) > 0:
            b_idx = b_idxs[0]
            points = np.array([lon_syn_tc[b_idx, :], lat_syn_tc[b_idx, :]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=wnd_cmap, norm=norm)
            lc.set_array(vmax_syn_tc[b_idx, :])
            lc.set_linewidth(5)
            line = ax.add_collection(lc)
            ax.scatter(lon_syn_tc[b_idx, 0], lat_syn_tc[b_idx, 0], s = 60, c = 'k')

    return fig

@app.route('/name', methods=['POST'])
def name():
    fn_ib = 'IBTrACS.ALL.v04r00.nc'
    ds_ib = xr.open_dataset(fn_ib)
    names = ds_ib['name'].data.astype('str')
    basins = ds_ib['basin'][:, 0].data.astype('str')
    r_name = request.form['name'].upper()

    if np.any(names == r_name):
        idxs = names == r_name

        if len(idxs) >= 0:
            lon_tc = ds_ib['lon'][idxs, :].load().data
            lat_tc = ds_ib['lat'][idxs, :].load().data
            basin_tc = basins[idxs]
            tc_names = names[idxs]
            vmax_tc = ds_ib['usa_wind'][idxs, :].load().data
            lon_syn_tc =np.array([])
            lat_syn_tc = np.array([])
            basin_syn_t = np.array([])
            vmax_syn_tc = np.array([]) 
            fig = plot_tracks(lon_tc, lat_tc, basin_tc, tc_names, vmax_tc,
                  lon_syn_tc, lat_syn_tc, basin_syn_t, vmax_syn_tc,r_name)


        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")

        # Embed the result in the html output.
        #data = base64.b64encode(buf.getbuffer()).decode("ascii")
        new_image_string = base64.b64encode(buf.getvalue()).decode("utf-8")
        return render_template('figure.html', img_data=new_image_string, h_name = '')
    else:
        return render_template('result.html', img_data='', h_name = 'No hurricanes found :(')

@app.route('/date', methods=['POST'])
def date():
    fn_ib = 'IBTrACS.ALL.v04r00.nc'
    ds_ib = xr.open_dataset(fn_ib)
    names = ds_ib['name'].load().data.astype('str')
    basins = ds_ib['basin'][:, 0].data.astype('str')

    try:
        r_date = request.form['date'].upper()
        r_date = datetime.datetime.strptime(r_date, '%Y-%m-%d')
    except:
        return render_template('result.html', img_data='', h_name = 'No hurricanes found :(')

    #ds_ipsl = xr.open_dataset('/home/clee/test/open_house/IPSL-CM6A-LR_openhouse.nc')
    ds_cesm = xr.open_dataset('/home/clee/test/open_house/CESM2_openhouse.nc')

    # For synthetic tracks, select a random year between 1951 and 2100
    # Use the month and day of the actual birthday.
    year_syn = np.random.randint(1951, 2101)
    cesm_mask = np.logical_and(ds_cesm['year'][0, :].data == year_syn,
                               ds_cesm['month'][0, :].data == r_date.month,
                               ds_cesm['day'][0, :].data == r_date.day)
    # 1 is Atlantic, 2 is eastern Pacific, 3 is western Pacific, 4 is northern
    # Indian Ocean, 5 is southern Indian Ocean, 6 is Australia, 7 is southern Pacific.
    basin_num_date = ds_cesm['basin'][0, cesm_mask].data
    basin_id_date = np.full(np.sum(cesm_mask), '').astype('object')
    basin_id_date[basin_num_date == 1] = 'NA'
    basin_id_date[basin_num_date == 2] = 'EP'
    basin_id_date[basin_num_date == 3] = 'WP'
    basin_id_date[basin_num_date == 4] = 'NI'
    basin_id_date[basin_num_date == 5] = 'SI'
    basin_id_date[basin_num_date == 6] = 'SP'
    basin_id_date[basin_num_date == 7] = 'SP'

    time = ds_ib['time'].load()
    null_mask = ~pd.isnull(time.data)
    dts = [datetime.datetime.utcfromtimestamp(int(x)/1e9) for x in np.array(time.data[null_mask])]
    dt = np.full(time.shape, datetime.datetime(2100, 1, 1))
    dt[null_mask] = dts
    tcs_close = np.abs(dt - r_date) <= datetime.timedelta(hours = 24)
    idxs = np.argwhere(np.any(tcs_close, axis = 1)).flatten()

    if len(idxs) > 0:
        lon_tc = ds_ib['lon'][idxs, :].load().data
        lat_tc = ds_ib['lat'][idxs, :].load().data
        basin_tc = basins[idxs]
        tc_names = names[idxs]
        vmax_tc = ds_ib['usa_wind'][idxs, :].load().data
    else:
        lon_tc = np.array([])
        lat_tc = np.array([])
        basin_tc = np.array([])
        tc_names = np.array([])
        vmax_tc = np.array([])
    # Synthetic tracks
    lon_syn_tc = []; lat_syn_tc = [];
    basin_syn_tc = []; vmax_syn_tc = [];
    for k in range(1, 8):
        idxs_syn_b = np.argwhere(basin_num_date == k).flatten()
        if len(idxs_syn_b) > 0:
            idx_syn_b = np.random.choice(idxs_syn_b)
            lon_fac = 0
            if basin_id_date[idx_syn_b] == 'NA' or basin_id_date[idx_syn_b] == 'EP':
                lon_fac = -360
            lon_syn_tc.append(ds_cesm['longitude'][:, cesm_mask][:, idx_syn_b].data + lon_fac)
            lat_syn_tc.append(ds_cesm['latitude'][:, cesm_mask][:, idx_syn_b].data)
            basin_syn_tc.append(basin_id_date[idx_syn_b])
            vmax_syn_tc.append(ds_cesm['Mwspd'][:, cesm_mask][:, idx_syn_b].data)
    lon_syn_tc = np.array(lon_syn_tc)
    lat_syn_tc = np.array(lat_syn_tc)
    basin_syn_tc = np.array(basin_syn_tc)
    vmax_syn_tc = np.array(vmax_syn_tc)
    if len(idxs) >= 0 :
        fig = plot_tracks(lon_tc, lat_tc, basin_tc, tc_names, vmax_tc,
                      lon_syn_tc, lat_syn_tc, basin_syn_tc, vmax_syn_tc,str(r_date).split(' ')[0])

        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")

        # Embed the result in the html output.
        #data = base64.b64encode(buf.getbuffer()).decode("ascii")
        new_image_string = base64.b64encode(buf.getvalue()).decode("utf-8")
        return render_template('figure.html', img_data=new_image_string, h_name = '')
    else:
        return render_template('result.html', img_data='', h_name = 'No hurricanes found :(')


class TC_Basin:
    """
    Base class that stores information of a basin.
    """

    def __init__(self, basin_id):
        if basin_id.upper() not in namelist.valid_basins:
            raise ValueError('Basin ID is not valid. See list of valid basins.')
        else:
            self.basin_id = basin_id
            self.basin_bounds = namelist.basin_bounds[self.basin_id]

    def _adj_bnd(self, bound):
        xd = float(bound[:-1])
        if bound[-1] in ['W', 'S']:
            xd *= -1
        return xd

    """
    Returns true if the position is within dx degrees of the basin bounds.
    """
    def in_basin(self, clon, clat, dx):
        lon_min, lat_min, lon_max, lat_max = self.get_bounds()

        is_in_basin = ((lon_min + dx) < clon < (lon_max - dx) and
                       (lat_min + dx) < clat < (lat_max - dx))
        return(is_in_basin)
    """
    Returns the lower left, and upper right coordinates of the longitude
    and latitude of the particular basin.
    """
    def get_bounds(self):
        bounds = self.basin_bounds

        ll_lon = self._adj_bnd(bounds[0])
        ll_lat = self._adj_bnd(bounds[1])
        ul_lon = self._adj_bnd(bounds[2])
        ul_lat = self._adj_bnd(bounds[3])

        return(ll_lon, ll_lat, ul_lon, ul_lat)

    """
    Reduces a global field to the basin boundaries.
    The global field has dimensions [latitude, longitude],
    which are described by lon, lat.
    """
    def transform_global_field(self, lon, lat, field):
        lon_min, lat_min, lon_max, lat_max = self.get_bounds()

        if lon[0] >= -1e-5 and (lon_min < 0 or lon_max < 0):
            # If the basin bounds are phrased in negative longitude
            # coordinates, transform the field into those coordinates.
            lon_t, X_t = self.transform_lon(lon, field)
        elif (lon < 0).any() and lon_min >= 0:
            # If the grid is phrased in negative longitude coordinates,
            # transform the grid into those coordinates.
            lon_t, X_t = self.transform_lon_r(lon, field)
        else:
            lon_t = lon
            X_t = field

        lon_mask = np.logical_and(lon_t <= (lon_max+1e-5), lon_t >= (lon_min - 1e-5))
        lat_mask = np.logical_and(lat >= (lat_min-1e-5), lat <= (lat_max + 1e-5))
        X_c = X_t[lat_mask, :]
        return (lon_t[lon_mask], lat[lat_mask], X_c[:, lon_mask])

    """
    Returns the basin array size.
    """
    def get_basin_size(self, lon, lat):
        lon_min, lat_min, lon_max, lat_max = self.get_bounds()
        if lon_min < 0 or lon_max < 0:
            lon_t, _ = self.transform_lon(lon, np.zeros((lat.size, lon.size)))
        else:
            lon_t = lon
        lon_mask = np.logical_and(lon_t <= lon_max, lon_t >= lon_min)
        lat_mask = np.logical_and(lat >= lat_min, lat <= lat_max)
        return (lat[lat_mask].size, lon_t[lon_mask].size)

    """
    Transform a field with longitude from 0-360E to -180-180.
    """
    def transform_lon(self, lon, X):
        lon_mask = lon >= (180 - 1e-5)
        X_t = np.concatenate((X[:, lon_mask],
                              X[:, np.logical_not(lon_mask)]), axis=1)
        lon_t = np.hstack((lon[lon_mask] - 360, lon[np.logical_not(lon_mask)]))
        return (lon_t, X_t)

    """
    Transform a field with longitude from -180-180 to 0-360E.
    """
    def transform_lon_r(self, lon, X):
        lon_mask = lon < -1e-5  # 0
        X_t = np.concatenate((X[:, np.logical_not(lon_mask)], X[:, lon_mask]), axis=1)
        lon_t = np.hstack((lon[np.logical_not(lon_mask)], lon[lon_mask] + 360))
        return (lon_t, X_t)
