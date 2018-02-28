from ktransit import FitTransit, FitTransitAutoGrad, LCModel, plot_results
import kplr
# import numpy as np
from autograd import numpy as np

def med_filt(x, y, dt=4.):
    """
    De-trend a light curve using a windowed median.

    """
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    assert len(x) == len(y)
    r = np.empty(len(y))
    for i, t in enumerate(x):
        inds = (x >= t - 0.5 * dt) * (x <= t + 0.5 * dt)
        r[i] = np.median(y[inds])
    return r

def norm_by_quarter(flux,ferr,quarter):
    for i in np.unique(quarter):
        ferr[quarter == i] /= np.median(
            flux[quarter == i])
        flux[quarter == i] /= np.median(
            flux[quarter == i])
    return flux, ferr


def byebyebaddata(flux,ferr,quality):
    finite = np.isfinite(flux)
    qflag = quality == 0
    mask = np.logical_and(finite,qflag)
    flux[~mask] = np.nan
    ferr[~mask] = np.nan
    return flux, ferr

def i_hate_nans(time,flux,ferr, quarter, quality):
    finite = np.isfinite(flux)
    return (time[finite],flux[finite],
            ferr[finite], quarter[finite], quality[finite])

def generate_data(kic=None, koi=None, do_fit_now=False):
    if koi is None and kic is None:
        raise Exception('Either `koi` or `kic` are required.')
    
    client = kplr.API()
    if kic is not None:
        star   = client.star(kic)
    if koi is not None:
        k0    = client.koi(koi)
        star  = k0.star
    
    lcs = star.get_light_curves(short_cadence=False)
    time, flux, ferr, quality,quarter = [], [], [], [], []
    for lc in lcs:
        with lc.open() as f:
            # The lightcurve data are in the first FITS HDU.
            hdu_data = f[1].data
            time = np.r_[time,hdu_data["time"]]
            #fluxval = hdu_data["sap_flux"]
            #fluxmed = np.median(fluxval)
            #flux = np.r_[flux,fluxval / fluxmed]
            #ferr = np.r_[ferr,hdu_data["sap_flux_err"] / fluxmed]
            flux = np.r_[flux,hdu_data["sap_flux"]]
            ferr = np.r_[ferr,hdu_data["sap_flux_err"]]
            quality = np.r_[quality,hdu_data["sap_quality"]]
            quarter = np.r_[quarter,f[0].header["QUARTER"] +
                np.zeros(len(hdu_data["time"]))]
    
    flux, ferr = byebyebaddata(flux,ferr,quality)
    time, flux, ferr, quarter, quality = i_hate_nans(time,
                                                     flux, ferr,
                                                     quarter, quality)
    flux, ferr = norm_by_quarter(flux, ferr,quarter)
    
    medfilt_dt1= med_filt(time,flux,dt=1.0)
    
    cflux = (flux / medfilt_dt1) - 1.0
    
    return time, cflux, flux, ferr, medfilt_dt1

def generate_fitT(time, flux, ferr, kic=None, koi=None, use_autograd=False, do_fit_now=False):
    if use_autograd:
        fitT = FitTransitAutoGrad()
    else:
        fitT = FitTransit()
    
    if koi is None and kic is None:
        raise Exception('Either `koi` or `kic` are required.')
    
    client = kplr.API()
    if kic is not None:
        star  = client.star(kic)
        k0    = star.kois[0]
    if koi is not None:
        k0    = client.koi(koi)
        star  = k0.star
    
    fitT.add_guess_star(rho=2.45)
    
    # fitT.add_guess_planet(
    #     period=k0.koi_period,
    #     impact=k0.koi_impact,
    #     T0=k0.koi_time0bk,
    #     rprs=k0.koi_ror)
    
    for koi_now in star.kois:
        try:
            fitT.add_guess_planet(
                period  = koi_now.koi_period,
                impact  = koi_now.koi_impact,
                T0      = koi_now.koi_time0bk,
                rprs    = koi_now.koi_ror)
        except:pass
    
    fitT.add_data(time=time,flux=flux,ferr=ferr)
    freeparstar = ['rho','zpt']
    freeparplanet = [
    'period','T0','impact','rprs']
    fitT.free_parameters(freeparstar,freeparplanet)
    
    if do_fit_now:
        fitT.do_fit()
    
    return fitT

if __name__ == '__main__':
    kic             = 8478994 # Kepler-37
    time, cflux, flux, ferr, medfilt_dt1  = generate_data(kic=kic)
    
    fitT  = generate_fitT(time,cflux,ferr, kic=kic)
    # time, cflux, flux, ferr, medfilt_dt1  = generate_data(koi=3.01)
    # fitT            = generate_fitT(time,cflux,ferr, koi=3.01)
    # fitT_ag         = generate_fitT(time,cflux,ferr, kic, use_autograd=True)
    
    fitT.do_fit()
    # fitT_ag.do_fit()
    
    fitT.print_results()
    # fitT_ag.print_results()
    
    fig     = plot_results(time,cflux,fitT.transitmodel)
    # fig_ag  = plot_results(time,cflux,fitT_ag.transitmodel)
    
    fig.savefig('ktransitfit.png')
    # fig_ag.savefig('ktransitfit_autograd.png')
