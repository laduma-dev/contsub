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
    cube = opts.image_name

    # get rid of stokes axis if it exists
    # TODO(sphe) Automate this
    # Needs to fixed later
    if opts.no_stokes is False:
        cube_slice = 0, slice(None), slice(None), slice(None), slice(None)
    else:
        cube_slice = slice(None)
        
    if len(opts.order) != len(opts.segments):
        raise RuntimeError("The --order and --segments lists must be the size.")
    niter = len(opts.segments)
        
    header = FitsHeader(cube[0].header)
    freqs = header.retFreq()         

    #get the mask for the first roun
    nomask = True
    if opts.mask_image:
        log.info("Loading mask image")
        mask = fitsio.getdata(opts.mask_image)[cube_slice]
        nomask = False
        
    else:
        #create a list of pixel sigma clip objects with the desired sigma clipping
        #create mask objects that corresponds to each pixel sigma clip object
        clips = [PixSigmaClip(ma) for ma in opts.sigma_clip]
        maskings = [Mask(clip) for clip in clips]
        mask = maskings[0].getMask(cube[0].data[0])

    #create a list of fit functions for each round
    #each round here is a spline 
    methds = [FitBSpline(*fa) for fa in zip(opts.order, opts.segments)]

    #run the first round of continuum subtraction
    #constsub = ContSub(freqs, cube[0].data[0], methds[0], None)
    #cont, line = constsub.fitContinuum()
    #log.info('finished fitting the first round')

    #run the next two rounds
    for i in range(niter):
        if nomask:
        #create mask
            fitfunc = FitBSpline(*[opts.order[i], opts.segments[i]])
            contsub = ContSub(freqs, )
            clip = PixSigmaClip(opts.sigma_clip[i])
            mask = Mask(clip).getMask(line)
            constsub = ContSub(freqs, cube[0].data[0], methds[i], mask)

        #do the fitting
        cont, line = constsub.fitContinuum()
        log.info(f'finished fitting round {i}')

    cube.close()

    #this uses a custom library that I wrote to save the line and continuum cubes
    cube = RCube('/scratch3/projects/laduma/ExpsAmir/completeness/run001_2deca_m/part022/laduma_dr1.1_image.1304~1420MHz.clean_pbcorr_pbuncorr_ch-01f01_y-04f06_x-04f06.fits')
    cube.openR()
    cube.write_like(cont, name = 'cont')
    cube.write_like(line, name = 'line')
    cube.close()
    
    
    
runit()