REGISTRY = {}

from .central_rnn_agent import CentralRNNAgent
from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["central_rnn"] = CentralRNNAgent