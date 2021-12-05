import logging
import sys

from . import grad_operators
from . import linear_operators
from . import map_uncertainty
from . import primal_dual
from . import prox_operators
from . import Empty

# create logger
logger = logging.getLogger("Optimus Primal")
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
