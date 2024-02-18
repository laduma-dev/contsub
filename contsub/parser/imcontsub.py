import contsub
from contsub.parser.utils import load, load_sources, File
from scabha.schema_utils import clickify_parameters
import click
from omegaconf import OmegaConf
import glob
import os

command = "imcontsub"
thisdir  = os.path.dirname(__file__)
source_files = glob.glob(f"{thisdir}/library/*.yaml")
sources = load_sources(source_files)
parserfile = File(f"{thisdir}/{command}.yaml")
config = load(parserfile, use_sources=sources)

@click.command(command)
@click.version_option(str(contsub.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    print(opts)
    
runit()