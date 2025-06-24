from .random_agent import RandomAgent
from .torch_agent import TorchAgent
from .sensory_neuron_agent import SensoryNeuronAgent

__all__ = ["RandomAgent",
           "TorchAgent",
           "SensoryNeuronAgent",
           ]

try:
    from .keyboard_agent import KeyboardAgent
    __all__ += ["KeyboardAgent"]

except ImportError: # XServer issue with pynput.keyboard
    pass
