#example contsub with this library

import sys
#append path to scripts to the PATH
sys.path.insert(1, "/scratch3/users/amirkazemim/pseudoCodes/pyScripts/contSub/")
from Cubes import RCube, FitsHeader
from astropy.io import fits
from Contsub import FitBSpline, ContSub, Mask, pixSigmaClip
import time

start_time = time.time()

#open the fits cube
cube = fits.open('/scratch3/projects/laduma/ExpsAmir/completeness/run001_2deca_m/part022/laduma_dr1.1_image.1304~1420MHz.clean_pbcorr_pbuncorr_ch-01f01_y-04f06_x-04f06.fits')

header = FitsHeader(cube[0].header)
freqs = header.retFreq()         

#create a list of pixel sigma clip objects with the desired sigma clipping
#here the first round is clipped at 4 sigma, second at 3.3 sigma and third at 2.6 sigma
clips = [pixSigmaClip(*ma) for ma in [[4], [3.3], [2.6]]]

#create mask objects that corresponds to each pixel sigma clip object
maskings = [Mask(clip) for clip in clips]

#get the mask for the first round
mask = maskings[0].getMask(cube[0].data[0])

#create a list of fit functions for each round
#each round here is a spline 
#first round is order 3 and segment width of 1750 km/s
#second round is order 3 and segment width of 1675 km/s
#third round is order 3 and segment width of 1500 km/s
methds = [FitBSpline(*fa) for fa in [[3, 1750], [3, 1675], [3, 1500]]]

#run the first round of continuum subtraction
constsub = ContSub(freqs, cube[0].data[0], methds[0], None)
cont, line = constsub.fitContinuum()
print('finished fitting the first round')

#run the next two rounds
for i in range(1, len(maskings)):
    del mask

    #create mask
    mask = maskings[i].getMask(line)
    constsub = ContSub(freqs, cube[0].data[0], methds[i], mask)
    del cont, line

    #do the fitting
    cont, line = constsub.fitContinuum()
    print(f'finished fitting round {i}')
    
cube.close()
print(f'fit_TIME - spent {time.time()-start_time:.1f} seconds for fitting the continuum')
start_time = time.time()

#this uses a custom library that I wrote to save the line and continuum cubes
cube = RCube('/scratch3/projects/laduma/ExpsAmir/completeness/run001_2deca_m/part022/laduma_dr1.1_image.1304~1420MHz.clean_pbcorr_pbuncorr_ch-01f01_y-04f06_x-04f06.fits')
cube.openR()
cube.write_like(cont, name = 'cont')
cube.write_like(line, name = 'line')
cube.close()
print(f'write_TIME - spent {time.time()-start_time:.1f} seconds for writing the continuum and line cube')

