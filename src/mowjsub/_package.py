"""Package-level constants, kept out of the package initializer.

An `__init__.py` that holds implementation is what RUF067
(`non-empty-init-module`) is about: importing the package then runs it, and
`from mowjsub import X` says nothing about where `X` is defined. Everything
lives here and the initializer re-exports it.
"""

import logging
from importlib import metadata
from types import SimpleNamespace

__version__ = metadata.version(__package__)

#: Command names. These used to be an OmegaConf object, which was a whole
#: config system for three constants.
BIN = SimpleNamespace(
    im_plane="im-mowjsub",
    vis_plane="vis-mowjsub",
    doppler_plane="doppler-mowjsub",
)

#: Every module logs to this one logger. It used to be one per entry point,
#: which meant `utils` -- shared by all three -- logged to the image-plane
#: logger whichever command you had actually run.
LOGGER = "mowjsub"

# Library convention, the same one shinobi keeps for `shinobi.*`: modules
# only ever *emit* through this logger and never attach a handler, so
# importing mowjsub configures nothing. The NullHandler is what makes that
# silence deliberate rather than logging's last-resort stderr echo of
# unhandled WARNING+ records. The console handler belongs to the
# application -- see `mowjsub.parser._cli`.
logging.getLogger(LOGGER).addHandler(logging.NullHandler())
