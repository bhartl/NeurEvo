""" Collection of `mindcraft.ReferenceFrame` implementations """

from .variance_loss import VarianceLoss
from .embedding_prediction import EmbeddingPrediction


__all__ = ["VarianceLoss",
           "EmbeddingPrediction",
           ]
