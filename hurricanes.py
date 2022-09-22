import base64
from io import BytesIO

from flask import Flask, render_template, request
from matplotlib.figure import Figure
import datetime
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', img_data='')

@app.route('/name', methods=['POST'])
def name():
    fn_ib = 'IBTrACS.NA.v04r00.nc'
    ds_ib = xr.open_dataset(fn_ib)
    names = ds_ib['name'].data.astype('str')
    r_name = request.form['name'].upper()
    if np.any(names == r_name):
        idxs = names == r_name

        lon_min = 260; lon_max = 360;
        lat_min = 0; lat_max = 60;
        lon = ds_ib['lon'][idxs, :].data
        lat = ds_ib['lat'][idxs, :].data

        lon_cen = 0
        dlon_label = 20
        lon_lmin = np.floor(lon_min / dlon_label) * dlon_label
        lon_lmax = np.ceil(lon_max / dlon_label) * dlon_label
        xlocs = np.arange(lon_lmin, lon_lmax + dlon_label, dlon_label)
        xlocs_shift = np.copy(xlocs)
        xlocs_shift[xlocs > lon_cen] -= 360

        # Generate the figure **without using pyplot**.
        fig = Figure()
        ax = fig.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        proj = ccrs.PlateCarree(central_longitude=lon_cen)
        ax = fig.add_subplot(111, projection=proj)
        ax.coastlines(resolution='50m')
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                     color='gray', alpha=0.3)
        gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                          color='gray', alpha=0.3)
        gl.xlabels_bottom = True
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        ax.plot(lon.T, lat.T);
        ax.set_xlim([-100, 0])
        fig.tight_layout()
        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")

        # Embed the result in the html output.
        #data = base64.b64encode(buf.getbuffer()).decode("ascii")
        new_image_string = base64.b64encode(buf.getvalue()).decode("utf-8")
        return render_template('figure.html', img_data=new_image_string, h_name = r_name)
    else:
        return render_template('index.html', img_data='')

@app.route('/date', methods=['POST'])
def date():
    fn_ib = 'IBTrACS.NA.v04r00.nc'
    ds_ib = xr.open_dataset(fn_ib)
    names = ds_ib['name'].data.astype('str')

    try:
        r_date = request.form['date'].upper()
        r_date = datetime.datetime.strptime(r_date, '%Y-%m-%d')
    except:
        return render_template('result.html', img_data='', h_name = 'No hurricanes found :(')

    null_mask = ~pd.isnull(ds_ib['time'].data)
    dts = [datetime.datetime.utcfromtimestamp(int(x)/1e9) for x in np.array(ds_ib['time'].data[null_mask])]
    dt = np.full(ds_ib['time'].shape, datetime.datetime(2100, 1, 1))
    dt[null_mask] = dts
    tcs_close = np.abs(dt - r_date) <= datetime.timedelta(hours = 24)
    idxs = np.argwhere(np.any(tcs_close, axis = 1)).flatten()

    if len(idxs) >= 0:
        lon_min = 260; lon_max = 360;
        lat_min = 0; lat_max = 60;
        lon = ds_ib['lon'][idxs, :].data
        lat = ds_ib['lat'][idxs, :].data

        lon_cen = 0
        dlon_label = 20
        lon_lmin = np.floor(lon_min / dlon_label) * dlon_label
        lon_lmax = np.ceil(lon_max / dlon_label) * dlon_label
        xlocs = np.arange(lon_lmin, lon_lmax + dlon_label, dlon_label)
        xlocs_shift = np.copy(xlocs)
        xlocs_shift[xlocs > lon_cen] -= 360

        # Generate the figure **without using pyplot**.
        fig = Figure()
        ax = fig.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        proj = ccrs.PlateCarree(central_longitude=lon_cen)
        ax = fig.add_subplot(111, projection=proj)
        ax.coastlines(resolution='50m')
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                     color='gray', alpha=0.3)
        gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                          color='gray', alpha=0.3)
        gl.xlabels_bottom = True
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        ax.plot(lon.T, lat.T);
        ax.set_xlim([-100, 0])
        fig.tight_layout()
        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")

        # Embed the result in the html output.
        #data = base64.b64encode(buf.getbuffer()).decode("ascii")
        new_image_string = base64.b64encode(buf.getvalue()).decode("utf-8")
        return render_template('figure.html', img_data=new_image_string, h_name = ', '.join(names[idxs]))
    else:
        return render_template('result.html', img_data='', h_name = 'No hurricanes found :(')
