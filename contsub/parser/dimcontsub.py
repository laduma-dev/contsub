import contsub
from scabha.schema_utils import clickify_parameters, paramfile_loader
import click
from scabha.basetypes import File
from omegaconf import OmegaConf
import glob
import os
from contsub import BIN
from scabha import init_logger
from contsub.imcontsub import FitBSpline, ContSub
import astropy.io.fits as fitsio
from contsub.utils import zds_from_fits, get_automask
import dask.array as da
import time
import numpy as np
import xarray as xr

log = init_logger(BIN.im_plane)

command = BIN.im_plane
thisdir  = os.path.dirname(__file__)
source_files = glob.glob(f"{thisdir}/library/*.yaml")
sources = [File(item) for item in source_files]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, sources)[command]


@click.command("dimcontsub")
@click.version_option(str(contsub.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    start_time = time.time()
 
    opts = OmegaConf.create(kwargs)
    infits = File(opts.input_image)
    
    if opts.output_prefix:
        prefix = opts.output_prefix
    else:
        prefix = f"{infits.BASEPATH}-contsub"
    
    outcont = File(f"{prefix}-cont.fits")
    outline = File(f"{prefix}-line.fits")
    if opts.overwrite is False and (outcont.EXISTS or outline.EXISTS):
        raise RuntimeError("At least one output file exists, but --no-overwrite has been set. Unset it to proceed.")
    
    if not infits.EXISTS:
        raise FileNotFoundError(f"Input FITS image could not be found at: {infits.PATH}")
    
    zds = zds_from_fits(infits)
    base_dims = ["ra", "dec", "spectral", "stokes"]
    if not hasattr(zds, "stokes"):
        base_dims.remove("stokes")
    
    dims_string = ",".join([f"{dim}" for dim in base_dims])
    signature = f"(spectral),({dims_string}),({dims_string}),(),() -> ({dims_string})"
    has_stokes = "stokes" in base_dims
    stokes_idx = opts.stokes_index
    if has_stokes:
        cube = zds.DATA[...,stokes_idx]
    else:
        cube = zds.DATA
    
    
    i = 0
    nworkers = 3
    sigma_clip = 5
    if opts.mask_image:
        mask_future = zds_from_fits(opts.mask_image).DATA
        nomask = False
    else:
        nomask = True
        
    if nomask and opts.automask:
        signature = f"(spectral),({dims_string}),(),(),() -> ({dims_string})"
        make_mask = da.gufunc(get_automask, 
                    signature = signature,
                    allow_rechunk = True,
                    output_dtypes=np.ndarray)
        
        mask_future = make_mask(
            zds.FREQS.data,
            cube.data,
            sigma_clip,
            opts.order[i],
            opts.segments[i],
    )
        nomask = False
    else:
        mask_future = da.zeros_like(cube, dtype=bool)

    
    signature = f"(spectral),({dims_string}),({dims_string}),() -> (spectral,dec,ra),(spectral,dec,ra)"
    
    fitfunc = FitBSpline(opts.order[i], opts.segments[i])
    contfit = ContSub(fitfunc, nomask=False, reshape=True, fitsaxes=False)
    
    meta = np.ndarray((), cube.dtype), np.ndarray((), cube.dtype)
    
    fitprods = xr.apply_ufunc(
        contfit.fitContinuum,
        zds.FREQS,
        cube,
        mask_future,
        input_core_dims = [ ("spectral",), base_dims, base_dims ],
        output_core_dims= [ ("spectral", "dec", "ra"), ("spectral", "dec", "ra")],
        dask='parallelized',
        dask_gufunc_kwargs = dict(meta=meta, allow_rechunk=True),
        
    )[0].compute(num_workers=5, shceduler="processes")
    
    header = zds.attrs["header"]
    
    log.info("Writing outputs") 
    
    if has_stokes:
        continuum = fitprods[0][...,np.newaxis]
        line = fitprods[1][...,np.newaxis]
    else:
        continuum = fitprods[0]
        line = fitprods[1]
    
    fitsio.writeto(outcont, continuum, header, overwrite=opts.overwrite)
    fitsio.writeto(outline, line, header, overwrite=opts.overwrite)

    # DONE
    dtime = time.time() - start_time
    hours = int(dtime/3600)
    mins = dtime/60 - hours*60
    log.info(f"Finished. Runtime {hours} hours and {mins:.2f} minutes")
