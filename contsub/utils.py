from xarrayfits import xds_from_fits
import xarray as xr
from astropy.wcs import WCS
from contsub.cubes import FitsHeader
from contsub.imcontsub import ContSub, Mask, PixSigmaClip, FitBSpline
from contsub import BIN
from scabha import init_logger
import astropy.io.fits as fitsio

log = init_logger(BIN.im_plane)

def get_automask(xspec, cube, sigma_clip=5, order=3, segments=400):

    log.info("Creating binary mask as requested") 
    fitfunc = FitBSpline(order,segments)
    contsub = ContSub(fitfunc, nomask=True, reshape=False, fitsaxes=False)
    _,line = contsub.fitContinuum(xspec, cube, mask=None)
    clip = PixSigmaClip(sigma_clip)
        
    mask = Mask(clip).getMask(line)
    log.info("Mask created suceesfully")
    return mask

def zds_from_fits(fname, chunks=None):
    """ Creates Zarr store from a FITS file. The resulting array has 
    dimensions = RA, DEC, SPECTRAL[, STOKES]

    Args:
        fname (str|path): FITS file_
        chunks (dict, optional): xarray chunk object. Defaults to {1: 25, 2:25}.

    Raises:
        RuntimeError: Input FITS file doesn't have a spectral axis
        FileNotFoundError: Input FITS file not found

    Returns:
        Zarr: Zarr array (persistant store, mode=w)
    """
    chunks = chunks or dict(ra=64,dec=None, spectral=None)
    fds = xds_from_fits(fname)[0]
    wcs = WCS(fds.hdu.header, naxis="spectral stokes".split())
    with fitsio.open(fname) as hdul:
        header = hdul[0].header

    axis_names = [header["CTYPE1"], header["CTYPE2"]] + wcs.axis_type_names
    if not wcs.has_spectral:
        raise RuntimeError("Input FITS file does not have a spectral axis")

    fds_xyz = fds.hdu.transpose(*axis_names)

    new_names = ["ra", "dec", "spectral"]
    if len(axis_names) == 4:
        new_names.append("stokes")

    coords = dict([(a,fds.hdu[b].values) for a,b in zip(new_names,axis_names)])
    ds = xr.Dataset(
        data_vars = dict(
            DATA = (new_names, fds_xyz.data),
            FREQS = (("spectral",), FitsHeader(header).retFreq()), 
            ),
        coords = coords,
        attrs = dict(
            info =f"Temporary copy of data from FITS file: {fname}",
            header = header,
                    ),
    )
    
    return ds.chunk(chunks)
    
#    return ds.to_zarr(store=f"{fname}.zarr", mode="w")
