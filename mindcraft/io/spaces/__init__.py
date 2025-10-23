""" Wrapper-module for `gym.spaces` with extended `Box` representable capabilities, (c) B. Hartl 2019 """
try:
    from gymnasium.spaces import *
except ImportError:
    from gym.spaces import *
from .box import Box
from .wrapper import space_clip
