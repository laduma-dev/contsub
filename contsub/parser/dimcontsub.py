import contsub
from scabha.schema_utils import clickify_parameters, paramfile_loader
import click
from scabha.basetypes import File
from omegaconf import OmegaConf
import glob
import os
from contsub import BIN
from contsub.cubes import RCube, FitsHeader
from contsub.imcontsub import FitBSpline, ContSub, Mask, PixSigmaClip
from scabha import init_logger
import astropy.io.fits as fitsio
from xarrayfits import xds_from_fits
import numpy as np

log = init_logger(BIN.im_plane)

command = BIN.im_plane
thisdir  = os.path.dirname(__file__)
source_files = glob.glob(f"{thisdir}/library/*.yaml")
sources = [File(item) for item in source_files]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, sources)[command]

@click.command(command)
@click.version_option(str(contsub.__version__))
@clickify_parameters(config)
def runit(**kwargs):
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
    # get rid of stokes axis if it exists
    # TODO(sphe) Automate this
    # Needs to be fixed later
    if opts.stokes_axis:
        dslice = 0, slice(None), slice(None), slice(None)
    else:
        dslice = slice(None)
        
    if len(opts.order) != len(opts.segments):
        raise RuntimeError("The --order and --segments lists must be the size.")
    niter = len(opts.segments)

    fits_dsl = xds_from_fits(infits)
    
    