""" Submodules comprising pytorch module fit to be used in neuro-evolution experiments """
from .patchwork import Patchwork
from .ensemble import Ensemble
from .complexify import Complexify
from .feed_forward import FeedForward
from .auto_encoder import AutoEncoder
from .convolution import Conv
from .convolution_transpose import ConvT
from .recurrent import Recurrent
from .projection import Projection
from .projection import LinearP
from .projection import PatchP
from .embedding import Embedding
from .embedding import StateEmbedding
from .embedding import GRNEmbedding
from .embedding import ConcatEmbedding
from .embedding import CPPNStateEmbedding
from .embedding import SensoryEmbedding
from .embedding import RegulatorEmbedding
from .set_transformer import SetTransformer
from .mixture_density_network import MixtureDensityNetwork
from .variance_covariance_embedding import VaCoEmbedding
from .neat_wiring import NEATWiring


__all__ = ['Patchwork',
           'Ensemble',
           'Complexify',
           'FeedForward',
           'AutoEncoder',
           'Conv',
           'ConvT',
           'Recurrent',
           'Projection',
           'LinearP',
           'PatchP',
           'Embedding',
           'StateEmbedding',
           'GRNEmbedding',
           'ConcatEmbedding',
           'SensoryEmbedding',
           'RegulatorEmbedding',
           'CPPNStateEmbedding',
           'SetTransformer',
           'MixtureDensityNetwork',
           'VaCoEmbedding',
           'NEATWiring',
           ]
