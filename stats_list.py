stats_list = [
    'mjd',           # mean MJD of scan
    'night',         # distance in h from 2 AM (UTC - 7)
    'az',            # mean azimuth of scan
    'el',            # mean elevation of scan
    'chi2',          # chi2 statistic for all timestreams of whole sb
    'acceptrate',    # acceptrate of sb
    'az_chi2',       # chi2 of azimuth binned timestreams
    'max_az_chi2',   # maximum chi2 of azimuth binned single frequency timestreams
    'med_az_chi2',   # median chi2 of azimuth binned single frequency timestreams
    'fbit',          # featurebit of scan (indicates scanning mode)
    'az_amp',        # average amplitude of fitted azimuth-template
    'el_amp',        # average amplitude of fitted elevation-template
    'n_spikes',      # number of spikes
    'n_jumps',       # number of jumps
    'n_anomalies',   # number of anomalies
    'tsys',          # average Tsys value of scan
    'pca1',          # average variance of removed pca mode 1
    'pca2',          # average variance of removed pca mode 2
    'pca3',          # average variance of removed pca mode 3
    'pca4',          # average variance of removed pca mode 4
    'weather',       # probability of bad weather
    'kurtosis',      # kurtosis of timestreams
    'skewness',      # skewness of timestreams
    'ps_chi2',       # (signed) chi2 of spherically averaged power spectrum for sb
    'scan_length',   # length of scan in minutes
    'saddlebag',     # saddlebag number (1-4)
    'sigma_poly0',   # sigma of mean in poly filter
    'fknee_poly0',   # fknee of mean in poly filter
    'alpha_poly0',   # alpha of mean in poly filter
    'sigma_poly1',   # sigma of slope in poly filter
    'fknee_poly1',   # fknee of slope in poly filter
    'alpha_poly1',   # alpha of slope in poly filter
    'power_mean',    # mean of sideband mean
    'sigma_mean',    # sigma of sideband mean
    'fknee_mean',    # fknee of sideband mean
    'alpha_mean',    # alpha of sideband mean
    'airtemp',       # hk: airtemp, C
    'dewtemp',       # hk: dewpoint temp, C
    'humidity',      # hk: relative humidity, (0-1)
    'pressure',      # hk: pressure, millibars
    'rain',          # hk: rain today, mm
    'winddir',       # hk: azimut from where wind is blowing, deg 
    'windspeed',     # hk: windspeed m/s
    'moon_dist',     # distance to the moon in deg
    'sun_dist'       # distance to the sun in deg
]