from __future__ import print_function
import pickle
import errno
import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import scipy.stats as stats
from scipy.optimize import curve_fit
from astropy.time import Time                                                                                                                                             
import astropy.coordinates as coord 
import astropy.units as u 
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris, EarthLocation                                                                                                    
from astropy.coordinates import get_body_barycentric, get_body, get_moon
import glob
import os
import pwd
import grp
import sys
import math
import multiprocessing
import importlib
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
os.environ["OMP_NUM_THREADS"] = "1"

class spike_data():
    def __init__(self):
        self.spike_types = ['spike', 'jump', 'anomaly', 'edge spike']
        pass

class spike_list():
    def __init__(self):
        self.spikes = []
        self.spike_types = ['spike', 'jump', 'anomaly', 'edge spike']
    
    def add(self, spike):
        self.spikes.append(spike)
    
    def addlist(self, sp_list):
        for sp in sp_list:
            self.add(sp)

    def sorted(self):
        lists = [[], [], []]
        for spike in self.spikes:
            lists[spike.type].append(spike)
        for typelist in lists:
            typelist.sort(key=lambda x: np.abs(x.amp), reverse=True)  # hat tip: https://stackoverflow.com/a/403426/5238625
        return lists

def get_spike_list(sb_mean, sd, scan_id, mjd):
    # cutoff = 0.0015 * 8.0
    my_spikes = spike_list()
    for spike_type in range(3):
        for spike in range(1000):
            sbs = sd[0, :, :, spike_type, spike]
            if np.all(sbs == 0):
                break
            max_sb = np.unravel_index(np.argmax(np.abs(sbs), axis=None), sbs.shape)
            max_ind = int(sd[1, max_sb[0], max_sb[1], spike_type, spike]) - 1
            s = spike_data()
            s.amp = sd[0, max_sb[0], max_sb[1], spike_type, spike]
            s.sbs = sbs
            s.ind = np.array((max_sb[0], max_sb[1], max_ind))  # feed, sb, ind
            s.mjd = mjd[max_ind]
            s.data = sb_mean[max_sb[0], max_sb[1], max_ind - 200:max_ind + 200]
            s.type = spike_type
            s.scanid = scan_id
            my_spikes.add(s)
    return my_spikes


def make_map(ra, dec, ra_bins, dec_bins, tod, mask):
    n_freq, n_samp = tod.shape

    # print(ra_bins)
    # print(dec_bins)
    # print(ra.shape)
    # print(dec.shape)
    n_pix_ra = len(ra_bins) - 1
    n_pix_dec = len(dec_bins) - 1
    map = np.zeros((n_pix_ra, n_pix_dec, n_freq))
    nhit = np.zeros_like(map)
    for i in range(n_freq):
        if mask[i] == 1.0:
            nhit[:, :, i] = np.histogram2d(ra, dec, bins=[ra_bins, dec_bins])[0]
            where = np.where(nhit[:, :, i] > 0)
            map[:, :, i][where] = np.histogram2d(ra, dec, bins=[ra_bins, dec_bins], weights=tod[i, :])[0][where] / nhit[:, :, i][where]
    return map, nhit


def compute_power_spec3d(x, k_bin_edges, dx=1, dy=1, dz=1):
    n_x, n_y, n_z = x.shape
    Pk_3D = np.abs(fft.fftn(x)) ** 2 * dx * dy * dz / (n_x * n_y * n_z)

    kx = np.fft.fftfreq(n_x, dx) * 2 * np.pi
    ky = np.fft.fftfreq(n_y, dy) * 2 * np.pi
    kz = np.fft.fftfreq(n_z, dz) * 2 * np.pi

    kgrid = np.sqrt(sum(ki ** 2 for ki in np.meshgrid(kx, ky, kz, indexing='ij')))

    Pk_nmodes = np.histogram(kgrid[kgrid > 0], bins=k_bin_edges, weights=Pk_3D[kgrid > 0])[0]
    nmodes = np.histogram(kgrid[kgrid > 0], bins=k_bin_edges)[0]

    k = (k_bin_edges[1:] + k_bin_edges[:-1]) / 2.0
    Pk = np.zeros_like(k)
    Pk[np.where(nmodes > 0)] = Pk_nmodes[np.where(nmodes > 0)] / nmodes[np.where(nmodes > 0)]
    return Pk, k, nmodes


def get_sb_ps(ra, dec, ra_bins, dec_bins, tod, mask, sigma, d_dec, n_k=10):
    map, nhit = make_map(ra, dec, ra_bins, dec_bins, tod, mask)
    h = 0.7
    deg2Mpc = 76.22 / h
    GHz2Mpc = 699.62 / h * (1 + 2.9) ** 2 / 115

    d_th = d_dec * deg2Mpc

    dz = 32.2e-3 * GHz2Mpc

    k_bin_edges = np.logspace(-1.8, np.log10(0.5), n_k)
    where = np.where(nhit > 0)
    rms = np.zeros_like(nhit)
    rms[where] = (sigma[None, None, :]/ np.sqrt(nhit))[where]
    w = np.zeros_like(nhit)
    w[where] = 1 / rms[where] ** 2
    # w = w / np.

    Pk, k, nmodes = compute_power_spec3d(w * map, k_bin_edges, d_th, d_th, dz)
    n_sim = 20
    ps_arr = np.zeros((n_sim, n_k - 1))
    for l in range(n_sim):
        map_n = np.random.randn(*rms.shape) * rms
        ps_arr[l] = compute_power_spec3d(w * map_n, k_bin_edges, d_th, d_th, dz)[0]
    
    transfer = 1.0 / np.exp((0.055/k) ** 2.5)  # 6.7e5 / np.exp((0.055/k) ** 2.5)#1.0 / np.exp((0.03/k) ** 2)   ######## Needs to be tested!
    
    ps_mean = np.mean(ps_arr, axis=0)
    ps_std = np.std(ps_arr, axis=0) / transfer
    Pk = Pk / transfer

    n_chi2 = len(k)
    chi = np.sum(((Pk - ps_mean)/ ps_std) ** 3)
    chi2 = np.sign(chi) * np.abs((np.sum(((Pk - ps_mean)/ ps_std) ** 2) - n_chi2) / np.sqrt(2 * n_chi2))
    return chi2, Pk, ps_mean, ps_std, transfer

# From Tony Li
def ensure_dir_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_params(param_file):
    params = {}
    with open(param_file) as f:
        fr = f.readlines()

        fr = [f[:] for f in fr]

        frs = [f.split(" = ") for f in fr]

        for stuff in frs:
            try:
                i, j = stuff
                params[str(i).strip()] = eval(j)
            except ValueError:
                pass
            except SyntaxError:
                if j == '.true.':
                    params[str(i).strip()] = True
                elif j == '.false.':
                    params[str(i).strip()] = False
                else:
                    pass
    return params


def read_runlist(params):
    filename = params['RUNLIST']

    with open(filename) as my_file:
        lines = [line.split() for line in my_file]
    i = 0

    fields = {}
    n_fields = int(lines[i][0])
    i = i + 1
    for i_field in range(n_fields):
        obsids = []
        scans = {}
        n_scans_tot = 0
        fieldname = lines[i][0]
        n_obsids = int(lines[i][1])
        print(fieldname)
        i = i + 1
        for j in range(n_obsids):
            obsid = lines[i][0]
            obsids.append(obsid)
            n_scans = int(lines[i][3])
            o_scans = []
            for k in range(1, n_scans - 1):
                if lines[i+k+1][0] != 8192:
                    o_scans.append(lines[i+k+1][0])
                    n_scans_tot += 1
            scans[obsid] = o_scans
            i = i + n_scans + 1 
        fields[fieldname] = [obsids, scans, n_scans_tot]
    return fields


def insert_data_in_array(data, indata, stats_string, obsid=False):
    try:
        index = stats_list.index(stats_string)
        if obsid:
            data[:, :, :, index] = indata
        else:
            data[:, :, index] = indata
    except ValueError:
        print('Did not find statistic "' + stats_string + '" in stats list.')


def extract_data_from_array(data, stats_string):
    try:
        index = stats_list.index(stats_string)
        outdata = data[:,:,:, index]
        return outdata
    except ValueError:
        print('Did not find statistic "' + stats_string + '" in stats list.')
        return 0

def get_scan_stats(filepath, map_grid=None):
    n_stats = len(stats_list)
    try:
        with h5py.File(filepath, mode="r") as my_file:
            tod_ind = np.array(my_file['tod'][:])
            n_det_ind, n_sb, n_freq, n_samp = tod_ind.shape
            sb_mean_ind = np.array(my_file['sb_mean'][:])
            point_tel_ind = np.array(my_file['point_tel'][:])
            point_radec_ind = np.array(my_file['point_cel'][:])
            mask_ind = my_file['freqmask'][:]
            mask_full_ind = my_file['freqmask_full'][:]
            reason_ind = my_file['freqmask_reason'][:]
            sigma0_ind = my_file['sigma0'][()]
            

            pixels = np.array(my_file['pixels'][:]) - 1 
            pix2ind = my_file['pix2ind'][:]
            scanid = my_file['scanid'][()]
            feat = my_file['feature'][()]

            airtemp = np.mean(my_file['hk_airtemp'][()])
            dewtemp = np.mean(my_file['hk_dewtemp'][()])
            humidity = np.mean(my_file['hk_humidity'][()])
            pressure = np.mean(my_file['hk_pressure'][()])
            rain = np.mean(my_file['hk_rain'][()])
            winddir = np.mean(my_file['hk_winddir'][()])
            windspeed = np.mean(my_file['hk_windspeed'][()])
            
            try:
                point_amp_ind = np.nanmean(my_file['el_az_stats'][()], axis=3) #### mean over chunk axis
            except:
                point_amp_ind = np.zeros((n_det_ind, n_sb, 1024, 2))
            try: 
                sd_ind = np.array(my_file['spike_data'])
            except KeyError:
                sd_ind = np.zeros((3, n_det_ind, n_sb, 4, 1000))
            try:
                tod_poly_ind = my_file['tod_poly'][()]
            except KeyError:
                tod_poly_ind = np.zeros((n_det_ind, n_sb, 2, n_samp))
            try: 
                chi2_ind = np.array(my_file['chi2'])
            except KeyError:
                chi2_ind = np.zeros_like(tod_ind[:,:,:,0])
            try:
                acc_ind = np.array(my_file['acceptrate'])
            except KeyError:
                acc_ind = np.zeros_like(tod_ind[:,:,0,0])
                print("Found no acceptrate")
            time = np.array(my_file['time'])
            mjd = time
            try:
                pca = np.array(my_file['pca_comp'])
                eigv = np.array(my_file['pca_eigv'])
                ampl_ind = np.array(my_file['pca_ampl'])
            except KeyError:
                pca = np.zeros((4, 10000))
                eigv = np.zeros(0)
                ampl_ind = np.zeros((4, *mask_full_ind.shape))
                print('Found no pca comps', scanid)
            try:
                tsys_ind = np.array(my_file['Tsys_lowres'])
            except KeyError:
                tsys_ind = np.zeros_like(tod_ind[:,:,:,0]) + 40
                print("Found no tsys")
    except KeyboardInterrupt:
        sys.exit()
    except:
        print('Could not load file', filepath, 'returning nans')
        data = np.zeros((20, 4, n_stats), dtype=np.float32)
        data[:] = np.nan
        return data

    t0 = time[0]
    time = (time - time[0]) * (24 * 60)  # minutes

    obsid = int(str(scanid)[:-2])

    n_freq_hr = len(mask_full_ind[0,0])
    n_det = 20

    data = np.zeros((n_det, n_sb, n_stats), dtype=np.float32)

    ## transform to full arrays with all pixels
    tod = np.zeros((n_det, n_sb, n_freq, n_samp))
    mask = np.zeros((n_det, n_sb, n_freq))
    mask_full = np.zeros((n_det, n_sb, n_freq_hr))
    acc = np.zeros((n_det, n_sb))
    ampl = np.zeros((4, n_det, n_sb, n_freq_hr))
    tsys = np.zeros((n_det, n_sb, n_freq))
    chi2 = np.zeros((n_det, n_sb, n_freq))
    sd = np.zeros((3, n_det, n_sb, 4, 1000))
    sb_mean = np.zeros((n_det, n_sb, n_samp))
    reason = np.zeros((n_det, n_sb, n_freq_hr))
    sigma0 = np.zeros((n_det, n_sb, n_freq))
    point_amp = np.zeros((n_det, n_sb, n_freq_hr, 2))
    tod_poly = np.zeros((n_det, n_sb, 2, n_samp))
    point_tel = np.zeros((n_det, n_samp, 3))
    point_radec = np.zeros((n_det, n_samp, 3))

    tod[pixels] = tod_ind
    mask[pixels] = mask_ind
    mask_full[pixels] = mask_full_ind
    reason[pixels] = reason_ind
    acc[pixels] = acc_ind
    ampl[:, pixels, :, :] = ampl_ind
    tsys[pixels] = tsys_ind
    chi2[pixels] = chi2_ind
    sd[:, pixels, :, :, :] = sd_ind
    sb_mean[pixels] = sb_mean_ind
    sigma0[pixels] = sigma0_ind
    point_amp[pixels] = point_amp_ind
    tod_poly[pixels] = tod_poly_ind
    point_tel[pixels] = point_tel_ind
    point_radec[pixels] = point_radec_ind

    az_amp = point_amp[:, :, :, 1]
    el_amp = point_amp[:, :, :, 0]


    mask_sum = mask_full.reshape((n_det, n_sb, n_freq, 16)).sum(3)
    az_amp = az_amp * mask_full
    el_amp = el_amp * mask_full

    az_amp_lowres = az_amp.reshape((n_det, n_sb, n_freq, 16)).sum(3) / mask_sum
    
    el_amp_lowres = el_amp.reshape((n_det, n_sb, n_freq, 16)).sum(3) / mask_sum

    mask_sb_sum = mask_full.sum(2)
    where = (mask_sb_sum > 0)
    az_amp_sb = np.zeros_like(mask_sb_sum)
    el_amp_sb = np.zeros_like(mask_sb_sum)
    az_amp_sb[where] = az_amp.sum(2)[where] / mask_sb_sum[where]
    el_amp_sb[where] = el_amp.sum(2)[where] / mask_sb_sum[where]

    my_spikes = get_spike_list(sb_mean, sd, str(scanid), mjd)
    sortedlists = my_spikes.sorted()
    n_spikes = len(sortedlists[0])
    n_jumps = len(sortedlists[1])
    n_anom = len(sortedlists[2]) 
    # cutoff = 0.0015 * 8.0
    n_sigma_spikes = 5         # Get from param file   ########################
    n_spikes_sb = (np.array([s.sbs for s in sortedlists[0]]) > 0.0015 * n_sigma_spikes).sum(0)
    n_jumps_sb = (np.array([s.sbs for s in sortedlists[1]]) > 0.0015 * n_sigma_spikes).sum(0)
    n_anom_sb = (np.array([s.sbs for s in sortedlists[2]]) > 0.0015 * n_sigma_spikes).sum(0)
    
    mask_sb_sum_lowres = mask.sum(2)
    tsys_sb = (tsys * mask).sum(2) / mask_sb_sum_lowres

    dt = (time[1] - time[0]) * 60  # seconds
    radiometer = 1 / np.sqrt(31.25 * 10 ** 6 * dt)
    ampl = np.abs(ampl).mean(3)
    ampl = 100 * np.sqrt(ampl ** 2 * pca.std(1)[:, None, None] ** 2 / radiometer ** 2)
    ampl[np.where(ampl == 0)] = np.nan

    # Here comes the different diagnostic data that is calculated

    # MJD
    scan_mjd = 0.5 * (mjd[0] + mjd[-1]) 
    insert_data_in_array(data, scan_mjd, 'mjd')

    # night
    hours = (scan_mjd * 24 - 7) % 24
    close_to_night = np.minimum(np.abs(2.0 - hours), np.abs(26.0 - hours))
    insert_data_in_array(data, close_to_night, 'night')

    # Mean az/el per feed
    mean_el = np.zeros((n_det, n_sb))
    mean_az = np.zeros((n_det, n_sb))

    mean_az[:, :] = np.mean(point_tel[:, :, 0], axis=1)[:, None]
    mean_el[:, :] = np.mean(point_tel[:, :, 1], axis=1)[:, None]

    insert_data_in_array(data, mean_az, 'az')
    insert_data_in_array(data, mean_el, 'el')

    # chi2 
    chi2_sb = np.sum(chi2, axis=2)
    n_freq_sb = np.nansum(mask, axis=2)
    wh = np.where(n_freq_sb != 0.0)
    chi2_sb[wh] = chi2_sb[wh] / np.sqrt(n_freq_sb[wh])
    wh = np.where(n_freq_sb == 0.0)
    chi2_sb[wh] = np.nan
    insert_data_in_array(data, chi2_sb, 'chi2')
 
    # acceptrate
    insert_data_in_array(data, acc, 'acceptrate')

    # azimuth binning
    nbins = 15                                ##### azimuth bins
    full_az_chi2 = np.zeros((n_det, n_sb))
    max_az_chi2 = np.zeros((n_det, n_sb))
    med_az_chi2 = np.zeros((n_det, n_sb))
    full_az_chi2[:] = np.nan
    max_az_chi2[:] = np.nan
    med_az_chi2[:] = np.nan
    az = point_tel[:, :, 0]
    for i in range(n_det):
        for j in range(n_sb):
            if acc[i, j]:
                freq_chi2 = np.zeros(n_freq)
                for k in range(n_freq): 
                    if mask[i, j, k] == 1.0:
                        histsum, bins = np.histogram(az[i], bins=nbins, weights=(tod[i, j, k]/sigma0[i,j,k]))
                        nhit = np.histogram(az[i], bins=nbins)[0]
                        normhist = histsum / nhit * np.sqrt(nhit)

                        # if i == 15 and j == 3 and k == 24:
                        #     print(scanid)
                        #     plt.errorbar(bins[1:], normhist, yerr=1/np.sqrt(nhit), fmt='-o')
                        #     plt.show()
                        freq_chi2[k] = (np.sum(normhist ** 2) - nbins) / np.sqrt(2 * nbins)
                        # if freq_chi2[k] > 4.0:
                        #     file = open('diag_az_bins.txt', 'a')
                        #     print(scanid, i+1, j+1, k+1, freq_chi2[k], scan_mjd, mean_az, mean_el, 
                        #           chi2[i,j,k], chi2_sb[i,j], tsys[i,j,k], feat, az_amp_lowres[i,j,k],
                        #           az_amp_sb[i,j], np.argmax(normhist ** 2),
                        #           file=file)
                        #     file.close()        
                full_az_chi2[i, j] = np.sum(freq_chi2) / np.sqrt(np.sum(mask[i,j]))
                max_az_chi2[i, j] = np.max(freq_chi2)
                med_az_chi2[i, j] = np.median(freq_chi2)
    insert_data_in_array(data, full_az_chi2, 'az_chi2')
    insert_data_in_array(data, max_az_chi2, 'max_az_chi2')
    insert_data_in_array(data, med_az_chi2, 'med_az_chi2')

    # featurebit
    insert_data_in_array(data, feat, 'fbit')

    # az-amplitude
    insert_data_in_array(data, az_amp_sb, 'az_amp')

    # el-amplitude
    insert_data_in_array(data, el_amp_sb, 'el_amp')

    # number of spikes, jumps, and anomalies
    insert_data_in_array(data, n_spikes_sb, 'n_spikes')
    insert_data_in_array(data, n_jumps_sb, 'n_jumps')
    insert_data_in_array(data, n_anom_sb, 'n_anomalies')

    # tsys averaged over sb
    insert_data_in_array(data, tsys_sb, 'tsys')

    # pca modes 
    insert_data_in_array(data, ampl[0], 'pca1')
    insert_data_in_array(data, ampl[1], 'pca2')
    insert_data_in_array(data, ampl[2], 'pca3')
    insert_data_in_array(data, ampl[3], 'pca4')

    # weather statistic
    try:
        weather  = np.loadtxt(weather_filepath)
        weather  = weather[np.where(np.isclose(obsid, weather))[0]]
        ten_min_in_mjd = 1 / 24.0 / 6.0

        i_start  = int((mjd[0] - weather[0, 3]) // ten_min_in_mjd)
        i_end    = int((mjd[-1] - weather[0, 3]) // ten_min_in_mjd)
        
        n_chunks = len(weather[:, 2])
        i_start  = min(i_start, n_chunks - 1)
        i_end    = min(i_end, n_chunks - 1)

        forecast = max(weather[i_start, 2], weather[i_end, 2])
    except IndexError:
        # no weather data for this obsid
        print('no weather data for obsid:', obsid)
        forecast = np.nan
    insert_data_in_array(data, forecast, 'weather')

    # add kurtosis etc of data histogram
    kurtosis = np.zeros((n_det, n_sb))
    skewness = np.zeros((n_det, n_sb))

    for i in range(n_det):
        for j in range(n_sb):
            if acc[i,j]:
                where = np.where(mask[i, j] > 0.0)

                normtod = (tod[i,j]/sigma0[i,j,:,None])[where].flatten()
                kurtosis[i,j] = stats.kurtosis(normtod)
                skewness[i,j] = stats.skew(normtod)

    insert_data_in_array(data, kurtosis, 'kurtosis')
    insert_data_in_array(data, skewness, 'skewness')

    # ps_chi2
    ra = point_radec[:, :, 0]
    dec = point_radec[:, :, 1]

    # centre = [(np.max(ra) + np.min(ra)) / 2, (np.max(dec) + np.min(dec)) / 2]

    # d_dec = 8.0 / 60 
    # d_ra = d_dec / np.cos(centre[1] / 180 * np.pi) # arcmin


    # n_pix = 16

    # ra_bins = np.linspace(centre[0] - d_ra * n_pix / 2, centre[0] + d_ra * n_pix / 2, n_pix + 1)
    # dec_bins = np.linspace(centre[1] - d_dec * n_pix / 2, centre[1] + d_dec * n_pix / 2, n_pix + 1)
    indices = np.zeros((n_det, 2, 2)).astype(int)
    ps_chi2 = np.zeros((n_det, n_sb))
    ps_chi2[:] = np.nan 
    map_list = [[None for _ in range(n_sb)] for _ in range(n_det)]
    for i in range(n_det):
        indices[i, 0, :] = np.digitize((np.min(ra[i]), np.max(ra[i])), map_grid[0])
        indices[i, 1, :] = np.digitize((np.min(dec[i]), np.max(dec[i])), map_grid[1])
        # print(indices[0])
        # print(ra)
        # print((np.min(ra[i]), np.max(ra[i])))
        # print((np.min(dec[i]), np.max(dec[i])))
        # print(map_grid)
        ra_bins = map_grid[0][indices[i, 0, 0] - 1:indices[i, 0, 1] + 1]
        dec_bins = map_grid[1][indices[i, 1, 0] - 1:indices[i, 1, 1] + 1]
        # print(indices[0])
        # print(map_grid)
        # print(ra_bins)
        # print(dec_bins)
        # sys.exit()
        for j in range(n_sb): ### should not need to be done per sideband.
            if acc[i, j]:
                map, nhit = make_map(ra[i], dec[i], ra_bins, dec_bins, tod[i, j], mask[i, j])
                where = np.where(nhit > 0)
                rms = np.zeros_like(nhit)
                rms[where] = (sigma0[i, j][None, None, :]/ np.sqrt(nhit))[where]
                map_list[i][j] = [map, rms]
                #ps_chi2[i, j], Pk, ps_mean, ps_std, transfer = get_sb_ps(ra, dec, ra_bins, dec_bins, tod[i, j], mask[i, j], sigma0[i, j], d_dec)
    
    #insert_data_in_array(data, ps_chi2, 'ps_chi2')
    
    # add length of scan
    duration = (mjd[-1] - mjd[0]) * 24 * 60  # in minutes
    insert_data_in_array(data, duration, 'scan_length')
    
    # saddlebags
    saddlebags = np.zeros((n_det, n_sb))
    saddlebags[(0, 3, 4, 11, 12), :] = 1  # feeds 1, 4, 5, 13, 14
    saddlebags[(5, 13, 14, 15, 16), :] = 2  # feeds 6, 14, 15, 16, 17
    saddlebags[(1, 6, 17, 18, 19), :] = 3  # feeds 2, 7, 18, 19, (20)
    saddlebags[(2, 7, 8, 9, 10), :] = 4  # feeds 3, 8, 9, 10, 11
    insert_data_in_array(data, saddlebags, 'saddlebag')
    
    # add one over f of polyfilter components
    sigma_poly = np.zeros((n_det, n_sb, 2))
    fknee_poly = np.zeros((n_det, n_sb, 2))
    alpha_poly = np.zeros((n_det, n_sb, 2))
    sigma_poly[:] = np.nan
    fknee_poly[:] = np.nan
    alpha_poly[:] = np.nan
    for i in range(n_det):
        for j in range(n_sb):
            if acc[i, j]:
                for l in range(2):
                    sigma_poly[i,j,l], fknee_poly[i,j,l], alpha_poly[i,j,l] = get_noise_params(tod_poly[i,j,l])
                    if np.isinf(sigma_poly[i,j,l]):
                        print('unable to fit noise params', scanid, i, j, l)
                    elif np.isnan(sigma_poly[i,j,l]):
                        print('nan in timestream', scanid, i, j, l)

    insert_data_in_array(data, sigma_poly[:,:,0], 'sigma_poly0')
    insert_data_in_array(data, fknee_poly[:,:,0], 'fknee_poly0')
    insert_data_in_array(data, alpha_poly[:,:,0], 'alpha_poly0')
    insert_data_in_array(data, sigma_poly[:,:,1], 'sigma_poly1')
    insert_data_in_array(data, fknee_poly[:,:,1], 'fknee_poly1')
    insert_data_in_array(data, alpha_poly[:,:,1], 'alpha_poly1')

    # sb_mean 
    power_mean = np.zeros((n_det, n_sb))
    sigma_mean = np.zeros((n_det, n_sb))
    fknee_mean = np.zeros((n_det, n_sb))
    alpha_mean = np.zeros((n_det, n_sb))
    power_mean[:] = np.nan
    sigma_mean[:] = np.nan
    fknee_mean[:] = np.nan
    alpha_mean[:] = np.nan
    for i in range(n_det):
        for j in range(n_sb):
            if acc[i, j]:
                power_mean[i,j] = np.mean(sb_mean[i,j])
                sigma_mean[i,j], fknee_mean[i,j], alpha_mean[i,j] = get_noise_params(sb_mean[i,j])
                if np.isinf(sigma_mean[i,j]):
                    print('unable to fit noise params', scanid, i, j)
                elif np.isnan(sigma_mean[i,j]):
                    print(np.argwhere(np.isnan(sb_mean[i,j])))
                    print('nan in timestream', scanid, i, j)


    insert_data_in_array(data, power_mean[:,:], 'power_mean')
    insert_data_in_array(data, sigma_mean[:,:], 'sigma_mean')
    insert_data_in_array(data, fknee_mean[:,:], 'fknee_mean')
    insert_data_in_array(data, alpha_mean[:,:], 'alpha_mean')

    # Housekeeping data
    insert_data_in_array(data, airtemp, 'airtemp')
    insert_data_in_array(data, dewtemp, 'dewtemp')
    insert_data_in_array(data, humidity, 'humidity')
    insert_data_in_array(data, pressure, 'pressure')
    insert_data_in_array(data, rain, 'rain')
    insert_data_in_array(data, winddir, 'winddir')
    insert_data_in_array(data, windspeed, 'windspeed')
    
    # sun and moon distances
    mean_ra = np.mean(ra)
    mean_dec = np.mean(dec)
    with solar_system_ephemeris.set('builtin'):   
        c = SkyCoord(ra=mean_ra*u.degree, dec=mean_dec*u.degree, frame='icrs')                                                                                                                            
        loc = coord.EarthLocation(lon=-118.283 * u.deg, lat=37.2313 * u.deg)
        t = Time(scan_mjd, format='mjd')
        moon = get_body('moon', t, loc)
        sun = get_body('sun', t, loc)
        cm = SkyCoord(ra=moon.ra.deg*u.degree, dec=moon.dec.deg*u.degree, frame='icrs')
        cs = SkyCoord(ra=sun.ra.deg*u.degree, dec=sun.dec.deg*u.degree, frame='icrs')
        d_moon = c.separation(cm).deg
        d_sun = c.separation(cs).deg
    insert_data_in_array(data, d_moon, 'moon_dist')
    insert_data_in_array(data, d_sun, 'sun_dist')


    ######## Here you can add new statistics  ##########

    return data, [map_list, indices]

def pad_nans(tod):
    n_pad = 10
    nan_indices = np.argwhere(np.isnan(tod))
    
    for nan_index in nan_indices:
        start_ind = int(max([nan_index - n_pad, 0]))
        end_ind = int(min(([nan_index + n_pad, len(tod) - 1])))
        mean = np.nanmean(tod[start_ind:end_ind])
        std = np.nanstd(tod[start_ind:end_ind])
        tod[nan_index] = mean + np.random.randn() * std
    return tod


def get_noise_params(tod, samprate=50.0):
    tod = tod[:-20]
    if not np.isfinite(np.mean(tod)):
        tod = pad_nans(tod)
    dt = 1 / samprate  # seconds

    n = len(tod)
    freq = fft.rfftfreq(n, dt)
    p = np.abs(fft.rfft(tod)) ** 2 / (n)
    bins = np.logspace(-2, 1, 20)
    nmodes = np.histogram(freq, bins=bins)[0]
    bin_freqs = np.histogram(freq, bins=bins, weights=freq)[0] / nmodes
    ps = np.histogram(freq, bins=bins, weights=p)[0] / nmodes

    sigma0 = np.std(tod[1:] - tod[:-1]) / np.sqrt(2)

    def one_over_f(freq, alpha, fknee):
        return sigma0 ** 2 * (1.0 + (freq / fknee) ** alpha)

    try: 
        p0 = (-2, 5)
        
        popt, pcov = curve_fit(one_over_f, bin_freqs, ps, p0=p0, sigma=ps/np.sqrt(nmodes))
        alpha = popt[0]
        fknee = popt[1]
    except:
        try:
            a = -1  # solve for alpha
            p0 = (a, 10)
            
            popt, pcov = curve_fit(one_over_f, bin_freqs, ps, p0=p0, sigma=ps/np.sqrt(nmodes))
            alpha = popt[0]
            fknee = popt[1]
        except: 
            return np.inf, np.inf, np.inf
    #     print('unable to fit noise parameters')
    #     print(ps)
    #     # return np.nan, np.nan, np.nan
    #     p0 = (-1, 10)
    #     plt.loglog(freq, p)
    #     plt.loglog(freq, sigma0 ** 2 + 0.0 * freq)
    #     plt.loglog(bin_freqs, ps) 
    #     plt.loglog(bin_freqs, one_over_f(bin_freqs, *p0), 'g--', label='fit: alpha=%5.3f, fknee=%5.3f' % tuple(p0))
    #     plt.show()
    #     popt, pcov = curve_fit(one_over_f, bin_freqs, ps, p0=p0, sigma=ps/np.sqrt(nmodes))
    #     sys.exit()
    #     # 
    return sigma0, fknee, alpha

class ObsidData():
    def __init__(self):
        pass

def get_scan_data(params, fields, fieldname, paralellize=False):
    l2_path = params['LEVEL2_DIR']
    field = fields[fieldname]
    n_scans = field[2]
    n_feeds = 20
    n_sb = 4
    n_stats = len(stats_list)
    
    scan_list = np.zeros((n_scans), dtype=np.int32)
    scan_data = np.zeros((n_scans, n_feeds, n_sb, n_stats), dtype=np.float32)
 
    if paralellize:    
        scanids = [scanid for obsid in field[0] for scanid in field[1][obsid]]

        filepaths = [l2_path + '/' + fieldname + '/' + fieldname + '_0' + scanid + '.h5' for scanid in scanids]
                                                                                                                                                                
        pool = multiprocessing.Pool(100)                                                                                                                              
        scan_data[:,:,:,:], _ = np.array(list(pool.map(get_scan_stats, filepaths)))
        scan_list[:] = scanids
    else:
        i_scan = 0
        for obsid in field[0]:
            scans = field[1][obsid]
            n_scans = len(scans)
            obsid_info = ObsidData()
            obsid_info.scans = scans
            obsid_info.field = fieldname
            obsid_info.l2_path = l2_path
            scan_data[i_scan:i_scan+n_scans] = get_obsid_data(obsid_info)
            scan_list[i_scan:i_scan+n_scans] = scans
            i_scan += n_scans
            # for scanid in scans:
            #     filepath = l2_path + '/' + fieldname + '/' + fieldname + '_0' + scanid + '.h5'
            #     #filepath = l2_path + '/' + fieldname + '_0' + scanid + '.h5'
            #     scan_data[i_scan] = get_scan_stats(filepath)
            #     scan_list[i_scan] = int(scanid)
            #     i_scan +=1
    return scan_list, scan_data


def get_obsid_data(obsid_info):
    scans = obsid_info.scans
    fieldname = obsid_info.field
    l2_path = obsid_info.l2_path
    
    ## set up map grid
    info = patch_info[fieldname]
    field_centre = np.array(info[:2]).astype(float)
    map_radius = int(info[2])  # degrees
    pix_side = int(info[4]) * 4  # 8 arcmin

    dx = np.linspace(-map_radius, map_radius, map_radius * 60 // pix_side + 1)
    ra = dx / np.cos(field_centre[1] * np.pi / 180) + field_centre[0]
    dec = dx + field_centre[1]

    map_grid = np.array([ra, dec])
    # print(map_grid)

    n_scans = len(scans)
    n_stats = len(stats_list)
    n_feeds = 20
    n_sb = 4
    scan_data = np.zeros((n_scans, n_feeds, n_sb, n_stats), dtype=np.float32)
    maps = []
    i_scan = 0
    for scanid in scans:
        filepath = l2_path + '/' + fieldname + '/' + fieldname + '_0' + scanid + '.h5'
        #filepath = l2_path + '/' + fieldname + '_0' + scanid + '.h5'
        data, map = get_scan_stats(filepath, map_grid)
        scan_data[i_scan] = data
        maps.append(map)
        i_scan += 1
    
    ps_s_sb_chi2, ps_s_feed_chi2, ps_s_chi2, ps_o_sb_chi2, ps_o_feed_chi2, ps_o_chi2 = get_power_spectra(maps, map_grid)

    insert_data_in_array(scan_data, ps_s_sb_chi2, 'ps_s_sb_chi2', obsid=True)
    insert_data_in_array(scan_data, ps_s_feed_chi2, 'ps_s_feed_chi2', obsid=True)
    insert_data_in_array(scan_data, ps_s_chi2, 'ps_s_chi2', obsid=True)
    insert_data_in_array(scan_data, ps_o_sb_chi2, 'ps_o_sb_chi2', obsid=True)
    insert_data_in_array(scan_data, ps_o_feed_chi2, 'ps_o_feed_chi2', obsid=True)
    insert_data_in_array(scan_data, ps_o_chi2, 'ps_o_chi2', obsid=True)

    ## [map_list, indices]
    # ## map_list[i][j] = [map, rms]
    # print('exiting')
    # sys.exit()
    return scan_data


def get_power_spectra(maps, map_grid):
    n_feeds = 20
    n_sb = 4
    n_k = 10
    n_scans = len(maps)
    ra, dec = map_grid
    h = 0.7
    deg2Mpc = 76.22 / h
    GHz2Mpc = 699.62 / h * (1 + 2.9) ** 2 / 115

    d_dec = dec[1] - dec[0]
    d_th = d_dec * deg2Mpc

    dz = 32.2e-3 * GHz2Mpc

    ps_s_sb_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    ps_s_feed_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    ps_s_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    # ps_s_stackp_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    # ps_s_stackfp_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    ps_o_sb_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    ps_o_feed_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    ps_o_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    # ps_o_stackp_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    # ps_o_stackfp_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    sum_obsid = np.zeros((len(ra) - 1, len(dec) - 1, n_sb, 64))  # np.zeros((len(ra), len(dec), 64))
    div_obsid = np.zeros_like(sum_obsid)
    sum_sb_obsid = np.zeros((n_feeds, len(ra) - 1, len(dec) - 1, n_sb, 64))  # np.zeros((len(ra), len(dec), 64))
    div_sb_obsid = np.zeros_like(sum_sb_obsid)
    ind_feed = []  # np.zeros((n_scans, n_feeds, 2, 2)).astype(int)
    accepted = np.zeros((n_scans, n_feeds, n_sb))
    for l in range(n_scans):  # need tests for if a scan is 
        map_list, indices = maps[l]
        if l == 0:
            pickle.dump(maps[l],open('scan1map.pkl','wb'))
            # np.save('scan1map.npy', maps[l])
        sum_scan = np.zeros((len(ra) - 1, len(dec) - 1, n_sb, 64))  # np.zeros((len(ra), len(dec), 64))
        div_scan = np.zeros_like(sum_scan)
        ind_feed.append(indices)
        for i in range(n_feeds):
            # ra_bins = map_grid[0][indices[i, 0, 0] - 1:indices[i, 0, 1] + 1]
            # dec_bins = map_grid[1][indices[i, 1, 0] - 1:indices[i, 1, 1] + 1]
            ind = indices[i]
            # print(indices)
            map_feed = np.zeros((ind[0, 1] - ind[0, 0] + 1, ind[1, 1] - ind[1, 0] + 1, n_sb, 64))  # np.zeros((len(ra), len(dec), 64))
            rms_feed = np.zeros_like(map_feed)
            for j in range(n_sb):
                if not map_list[i][j]:
                    ps_s_sb_chi2[l, i, j] = np.nan
                else:
                    accepted[l, i, j] = 1.0
                    map, rms = map_list[i][j]    ########### flip frequencies!! ##############
                    
                    
                    # print(map[:,:,10])
                    # print(rms[:,:,10])
                    ps_s_sb_chi2[l, i, j] = get_ps_chi2(map, rms, n_k, d_th, dz)  # , Pk, ps_mean, ps_std, transfer 
                    chi2 = ps_s_sb_chi2[l, i, j]
                    if np.isnan(chi2):
                        print(map[np.isnan(map)])
                        print(rms[np.isnan(rms)])
                        sys.exit()
                    map_feed[:, :, j, :] = map
                    rms_feed[:, :, j, :] = rms
                    
                    where = np.where(rms > 0)
                    sum_sb_obsid[i, ind[0, 0] - 1:ind[0, 1], ind[1, 0] - 1:ind[1, 1], j, :][where] += map[where] / rms[where] ** 2
                    div_sb_obsid[i, ind[0, 0] - 1:ind[0, 1], ind[1, 0] - 1:ind[1, 1], j, :][where] += 1.0 / rms[where] ** 2 
            if np.sum(accepted[l, i, :]) == 0:
                ps_s_feed_chi2[l, i, :] = np.nan
            else:
                if ((l == 0) and (i == 0)):
                    np.save('map_feed.npy', map_feed)
                    np.save('rms_feed.npy', rms_feed)

                sh = map_feed.shape
                ps_s_feed_chi2[l, i, :] = get_ps_chi2(
                    map_feed.reshape((sh[0], sh[1], n_sb * 64)),
                    rms_feed.reshape((sh[0], sh[1], n_sb * 64)),
                    n_k, d_th, dz, is_feed=True)
                where = np.where(rms_feed > 0.0)
                sum_scan[indices[i, 0, 0] - 1:indices[i, 0, 1], indices[i, 1, 0] - 1:indices[i, 1, 1], :, :][where] += map_feed[where] / rms_feed[where] ** 2 
                div_scan[indices[i, 0, 0] - 1:indices[i, 0, 1], indices[i, 1, 0] - 1:indices[i, 1, 1], :, :][where] += 1.0 / rms_feed[where] ** 2 
        if np.sum(accepted[l, :, :].flatten()) == 0:
            ps_s_chi2[l, :, :] = np.nan
        else:        
            map_scan = np.zeros_like(sum_scan)
            rms_scan = np.zeros_like(sum_scan)
            where = np.where(div_scan > 0.0)
            map_scan[where] = sum_scan[where] / div_scan[where]
            rms_scan[where] = np.sqrt(1.0 / div_scan[where])
            sh = map_scan.shape
            map_scan = map_scan.reshape((sh[0], sh[1], n_sb * 64))
            rms_scan = rms_scan.reshape((sh[0], sh[1], n_sb * 64))
            indices = np.ma.masked_equal(indices, 0, copy=False).astype(int)
            min_ind = np.min(indices[:, :, 0], axis=0)  ## only use the non-masked sidebands
            max_ind = np.max(indices[:, :, 1], axis=0)
            # print(min_ind, max_ind)
            # print(map_scan.shape)
            ps_s_chi2[l, :, :] = get_ps_chi2(
                    map_scan[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]],
                    rms_scan[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]],
                    n_k, d_th, dz, is_feed=True)

            sum_obsid[where] += sum_scan[where]
            div_obsid[where] += div_scan[where]
    if np.sum(accepted[:, :, :].flatten()) == 0:
        ps_o_sb_chi2[:] = np.nan
        ps_o_feed_chi2[:] = np.nan
        ps_o_chi2[:] = np.nan
    else:
        map_obsid = np.zeros_like(sum_obsid)
        rms_obsid = np.zeros_like(sum_obsid)
        where = np.where(div_obsid > 0.0)
        map_obsid[where] = sum_obsid[where] / div_obsid[where]
        rms_obsid[where] = np.sqrt(1.0 / div_obsid[where])
        sh = map_obsid.shape
        map_obsid = map_obsid.reshape((sh[0], sh[1], n_sb * 64))
        rms_obsid = rms_obsid.reshape((sh[0], sh[1], n_sb * 64))

        ind_feed = np.array(ind_feed).astype(int)
        ind_feed = np.ma.masked_equal(ind_feed, 0, copy=False).astype(int)
        min_ind = np.min(ind_feed[:, :, :, 0], axis=(0, 1))
        max_ind = np.max(ind_feed[:, :, :, 1], axis=(0, 1))
        ps_o_chi2[:, :, :] = get_ps_chi2(
                    map_obsid[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]],
                    rms_obsid[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]],
                    n_k, d_th, dz, is_feed=True)
        # sum_sb_obsid = np.zeros((n_feeds, len(ra), len(dec), n_sb, 64)) 
        for i in range(n_feeds):
            min_ind = np.min(ind_feed[:, i, :, 0], axis=0)
            max_ind = np.max(ind_feed[:, i, :, 1], axis=0)
                
            for j in range(n_sb):
                if np.sum(accepted[:, i, j]) == 0:
                    ps_o_sb_chi2[:, i, j] = np.nan
                else: 
                    map_sb = np.zeros((len(ra), len(dec), 64))
                    rms_sb = np.zeros((len(ra), len(dec), 64))
                    where = np.where(div_sb_obsid[i, :, :, j, :] > 0)
                    map_sb[where] = sum_sb_obsid[i, :, :, j, :][where] / div_sb_obsid[i, :, :, j, :][where]
                    rms_sb[where] = np.sqrt(1.0 / div_sb_obsid[i, :, :, j, :][where])
                    ps_o_sb_chi2[:, i, j] = get_ps_chi2(
                        map_sb[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]],
                        rms_sb[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]],
                        n_k, d_th, dz)
            if np.sum(accepted[:, i, :].flatten()) == 0:
                ps_o_feed_chi2[:, i, :] = np.nan
            else:   
                map_feed = np.zeros((len(ra), len(dec), n_sb, 64)) #np.zeros((max_ind[0] - min_ind[0] + 2, max_ind[1] - min_ind[1] + 2, n_sb, 64)) #np.zeros((len(ra), len(dec), n_sb, 64))        
                rms_feed = np.zeros_like(map_feed)
                where = where = np.where(div_sb_obsid[i, :, :, :, :] > 0)
                map_feed[where] = sum_sb_obsid[i, :, :, :, :][where] / div_sb_obsid[i, :, :, :, :][where]
                rms_feed[where] = np.sqrt(1.0 / div_sb_obsid[i, :, :, :, :][where])
                map_feed = map_feed[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]]
                rms_feed = rms_feed[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]]
                sh = map_feed.shape
                ps_o_feed_chi2[:, i, :] = get_ps_chi2(
                        map_feed.reshape((sh[0], sh[1], n_sb * 64)),
                        rms_feed.reshape((sh[0], sh[1], n_sb * 64)),
                        n_k, d_th, dz, is_feed=True)
            
    return (ps_s_sb_chi2, ps_s_feed_chi2, ps_s_chi2, ps_o_sb_chi2,
            ps_o_feed_chi2, ps_o_chi2)
                # return (ps_s_sb_chi2, ps_s_feed_chi2, ps_s_chi2, ps_s_stackp_chi2, ps_s_stackfp_chi2, ps_o_sb_chi2,
    #         ps_o_feed_chi2, ps_o_chi2, ps_o_stackp_chi2, ps_o_stackfp_chi2)

def get_ps_chi2(map, rms, n_k, d_th, dz, is_feed=False):

    where = np.where(rms > 0)
    k_bin_edges = np.logspace(-1.8, np.log10(0.5), n_k)
    w = np.zeros_like(rms)
    w[where] = 1 / rms[where] ** 2

    Pk, k, nmodes = compute_power_spec3d(w * map, k_bin_edges, d_th, d_th, dz)

    where = np.where(Pk > 0)

    n_sim = 20
    ps_arr = np.zeros((n_sim, n_k - 1))
    for l in range(n_sim):
        map_n = np.random.randn(*rms.shape) * rms
        ps_arr[l] = compute_power_spec3d(w * map_n, k_bin_edges, d_th, d_th, dz)[0]

    transfer = 1.0 / np.exp((0.055/k) ** 2.5)  # 6.7e5 / np.exp((0.055/k) ** 2.5)#1.0 / np.exp((0.03/k) ** 2)   ######## Needs to be tested!
    
    ps_mean = np.mean(ps_arr, axis=0)

    if is_feed:
        # transfer4 = 1.0 / np.exp((0.050/k) ** 5.5)  + 1e-6 
        transfer = np.array([7.08265320e-07, 1.30980902e-06, 1.87137602e-01, 4.91884922e-01, 6.48433271e-01, 8.27296733e-01, 8.85360854e-01, 8.14043197e-01, 8.03513664e-01]) #1.0 / np.exp((0.050/k) ** 5.5) + 1e-6
        with open("feed_ps.txt", "ab") as myfile:
            np.savetxt(myfile, np.array([Pk, ps_mean]).T)

    ps_std = np.std(ps_arr, axis=0) / transfer
    Pk = Pk / transfer

    n_chi2 = len(Pk[where])
    if n_chi2 < 5:
        return np.nan
    chi = np.sum(((Pk[where] - ps_mean[where])/ ps_std[where]) ** 3)
    chi2 = np.sign(chi) * np.abs((np.sum(((Pk[where] - ps_mean[where])/ ps_std[where]) ** 2) - n_chi2) / np.sqrt(2 * n_chi2))

    # if chi2 < -20.0:
    #     plt.errorbar(k, Pk * transfer, ps_std)
    #     plt.loglog(k, transfer * ps_mean[-1])
    #     plt.loglog(k, ps_mean * transfer)
    #     print(k, ps_std * transfer)
    #     plt.show()
    return chi2 #, Pk, ps_mean, ps_std, transfer



def get_patch_info(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        my_list = [[x for x in line.split()] for line in lines]
        my_dict = { y[0]: y[1:7] for y in my_list[:-1] }
    return my_dict

def save_data_2_h5(params, scan_list, scan_data, fieldname):
    filename = data_folder + 'scan_data_' + id_string + fieldname + '.h5'
    f1 = h5py.File(filename, 'w')
    f1.create_dataset('scan_list', data=scan_list)
    f1.create_dataset('scan_data', data=scan_data)
    f1.close()


def make_accept_list(params, accept_params, scan_data):
    n_scans, n_det, n_sb, _ = scan_data.shape
    accept_list = np.ones((n_scans, n_det, n_sb), dtype=np.bool)

    # decline all sidebands that are entirely masked
    acceptrate = extract_data_from_array(scan_data, 'acceptrate')

    # accept_list[:, 7, :] = False

    acc = np.zeros(len(stats_list) + 1)
    acc[0] = np.nansum(acceptrate[accept_list]) / (n_scans * 19 * 4)

    for i, stat_string in enumerate(stats_list):
        stats = extract_data_from_array(scan_data, stat_string)
        cuts = accept_params.stats_cut[stat_string]
        accept_list[np.where(np.isnan(stats))] = False
        accept_list[np.where((stats < cuts[0]) + (stats > cuts[1]))] = False
        acc[i+1] = np.nansum(acceptrate[accept_list]) / (n_scans * 19 * 4)
        print(acc[i+1], stat_string)

    return accept_list, acc


def make_jk_list(params, accept_list, scan_list, scan_data):
    n_scans, n_det, n_sb, _ = scan_data.shape
    jk_list = np.zeros((n_scans, n_det, n_sb), dtype=np.int32)

    if not np.any(accept_list.flatten()):
        jk_list[:] = 0 
        return jk_list

    
    # even/odd 
    obsid = [int(str(scanid)[:-2]) for scanid in scan_list]
    odd = np.array(obsid) % 2

    jk_list[:] += odd[:, None, None] * 2  # 2^1


    # day/night split
    mjd = extract_data_from_array(scan_data, 'mjd')
    hours = (mjd * 24 - 7) % 24
    closetonight = np.minimum(np.abs(2.0 - hours), np.abs(26.0 - hours))
    
    cutoff = np.percentile(closetonight[accept_list], 50.0)
    jk_list[np.where(closetonight > cutoff)] += 4 # 2^2

    # halfmission split
    cutoff = np.percentile(mjd[accept_list], 50.0)
    jk_list[np.where(mjd > cutoff)] += 8 # 2^3

    # saddlebag split
    saddlebags = extract_data_from_array(scan_data, 'saddlebag')
    jk_list[np.where(saddlebags > 2.5)] += 16 # 2^4


    ######## Here you can add new jack-knives  ############


    # insert 0 on rejected sidebands, add 1 on accepted 
    jk_list[np.invert(accept_list)] = 0
    jk_list[accept_list] += 1 

    return jk_list


def save_jk_2_h5(params, scan_list, acceptrates, accept_list, jk_list, fieldname): 
    filename = data_folder + 'jk_data_' + id_string + fieldname + '.h5'
    f1 = h5py.File(filename, 'w')
    f1.create_dataset('scan_list', data=scan_list)
    f1.create_dataset('acceptrates', data=acceptrates)
    f1.create_dataset('accept_list', data=accept_list)
    f1.create_dataset('jk_list', data=jk_list)
    f1.close()


if __name__ == "__main__":
    try:
        param_file = sys.argv[1]
    except IndexError:
        print('You need to provide param file as command-line argument')
        sys.exit()

    params = get_params(param_file)
    sys.path.append(params['ACCEPT_PARAM_FOLDER'])
    accept_params = importlib.import_module(params['ACCEPT_MOD_PARAMS'][:-3])
    stats_list = importlib.import_module(params['STATS_LIST'][:-3]).stats_list
    fields = read_runlist(params)

    patch_filepath = params['PATCH_DEFINITION_FILE']
    patch_info = get_patch_info(patch_filepath)

    weather_filepath = params['WEATHER_FILEPATH']
    data_folder = params['ACCEPT_DATA_FOLDER']
    id_string = params['ACCEPT_DATA_ID_STRING'] + '_'
    if id_string == '_':
        id_string = ''
    data_from_file = params['SCAN_STATS_FROM_FILE'] # False #True

    show_plot = params['SHOW_ACCEPT_PLOT']

    for fieldname in fields:
        if data_from_file:
            filepath = data_folder + 'scan_data_' + id_string + fieldname + '.h5'
            with h5py.File(filepath, mode="r") as my_file:
                scan_list = my_file['scan_list'][()]
                scan_data = my_file['scan_data'][()]
        else:
            scan_list, scan_data = get_scan_data(params, fields, fieldname)
        save_data_2_h5(params, scan_list, scan_data, fieldname)
        accept_list, acc = make_accept_list(params, accept_params, scan_data)
        jk_list = make_jk_list(params, accept_list, scan_list, scan_data)
 
        save_jk_2_h5(params, scan_list, acc, accept_list, jk_list, fieldname)

        if show_plot:
            labels = ['freq'] + stats_list 
            ind = np.arange(len(acc))
            plt.bar(ind, acc * 19, alpha=0.5, label=fieldname)
            plt.xticks(ind, labels, rotation='vertical')
    if show_plot:
        plt.ylabel('effective # of feeds')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


#### TODO:
## add elevation of individual feeds
