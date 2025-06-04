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
import numpy as np
import time

log = init_logger(BIN.im_plane)

command = BIN.im_plane
thisdir  = os.path.dirname(__file__)
source_files = glob.glob(f"{thisdir}/library/*.yaml")
sources = [File(item) for item in source_files]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, sources)[command]

start_time = time.time()
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
    # Needs to fixed later
    if opts.stokes_axis:
        dslice = 0, slice(None), slice(None), slice(None)
    else:
        dslice = slice(None)
        
    if len(opts.order) != len(opts.segments):
        raise RuntimeError("The --order and --segments lists must be the size.")
    niter = len(opts.segments)
    # get Fits image cube primary HDU
    with fitsio.open(opts.input_image) as hdul:
        header = hdul[0].header
        cube = hdul[0].data[dslice]
        freqs = FitsHeader(header).retFreq()         
    
    #get the mask for the first round
    nomask = True
    if opts.mask_image:
        log.info("Loading mask image")
        mask = fitsio.getdata(opts.mask_image)[dslice]
        mask_isnan = np.isnan(mask)
        
        if mask_isnan.any():
            mask[mask_isnan] = 0
            mask[~mask_isnan] = 1
        else:
            del mask_isnan
            mask = ~np.array(mask, dtype=bool)
        nomask = False
    
    prev_sclip = opts.sigma_clip[0]
    sigma_clip = list(opts.sigma_clip)
    for i in range(niter):
        
        fitfunc = FitBSpline(*[opts.order[i], opts.segments[i]])
        if nomask:
            if i == 0:
                log.info(f'Creating initial mask from input image')
                contsub = ContSub(freqs, cube, fitfunc)
                cont, line = contsub.fitContinuum()
            
            #create mask from line emission of first iteration
            try:
                sclip = sigma_clip[i]
            except IndexError:
                sclip = prev_sclip
            finally:
                prev_sclip = sclip
                
            clip = PixSigmaClip(sclip)
            mask = Mask(clip).getMask(line)
            
        log.info(f'Running iteration {i+1}')
        constsub = ContSub(freqs, cube, fitfunc, mask)
        #do the fitting
        cont, line = constsub.fitContinuum()
    log.info("Continuum fitting successful. Ready to write output products.")
        
    del cube
        
    
    log.debug(f'Pixel sums line: {line.sum()}, cont: {cont.sum()}')
    if opts.stokes_axis:
        cont = cont[np.newnaxis,...]
        line = line[np.newnaxis,...]
    log.info("Writing outputs") 
    fitsio.writeto(outcont, cont, header, overwrite=opts.overwrite)
    fitsio.writeto(outline, line, header, overwrite=opts.overwrite)

    # DONE
    dtime = time.time() - start_time
    hours = int(dtime/3600)
    mins = dtime/60 - hours*60
    log.info(f"Finished. Runtime {hours} hours and {mins:.2f} minutes")
    