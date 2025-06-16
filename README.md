# contsub
Image-Plane continuum subtraction for FITS cubes

## Installation
### PyPI (stable; recommended)
```bash
pip install contsub

```
### Latest (in development)
```bash
pip install git+https://github.com/laduma-dev/contsub.git
```



## Documentation
```bash
Usage: imcontsub [OPTIONS] File

Options:
  --version                       Show the version and exit.
  --order int,int,...             Order of spline. If given as a list of size
                                  N, then N iterations will be perfomed.
  --segments float,float,...      Width of spline segments in km/s. If given
                                  as a list, then it must have same sixe as
                                  --order.
  --output-prefix str             Name of ouput image
  --mask-image File               Mask image
  --sigma-clip float,float,...    Sigma clip for each iteration. Only required
                                  if mask-image is not given.
  --fit-model str                 Fit function to model the continuum. ** Only
                                  the spline is available in this version **.
  --cont-fit-tol float            Minimum perentage of valid spectrum data
                                  points required to do a fit. Spectra below
                                  this tolerance will be set to NaN. Leaving
                                  this unset may result in poor or NaN spectra
                                  in the output cubes
  --overwrite / --no-overwrite    Overwrite output image if it already exists
  --stokes-index int              ### NOT IMPLEMENTED. Stokes index 0 is
                                  assumed ### Index of stokes channel (zero-
                                  based) to use.
  --stokes-axis / --no-stokes-axis
                                  ### DEPRECATED #### Set this flag if the
                                  input image has a stokes dimension. (Default
                                  is True).
  --ra-chunks int                 Chunking along RA-axis. If set to zero, no
                                  Chunking is perfomed.
  --nworkers int                  Number of workers (one per CPU)
  --help                          Show this message and exit.

```
