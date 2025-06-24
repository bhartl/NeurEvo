import torch
import numpy as np
from mindcraft.io.spaces import Box, Discrete


def space_clip(value: (float, int, np.ndarray), space_obj: (Box, Discrete, tuple)):
    """ clips `value` to `space_obj` boundaries (for either Box or Discrete mindcraft.io.spaces.Space)

    - In the case of a Discrete space, 0 and (n - 1) are considered as boundaries
    - In the case of a Box space, low and high are considered as boundaries

    :param value: Number or np.ndarray, which is to be clipped to the `space_obj` boundaries
    :param space_obj: Box or Discrete mindcraft.io.spaces.Space instances or tuple, specifying lower and upper bound.
    :returns: values clipped to space boundaries.
              In case of a Discrete space, the clipping is performed to rounded values
              and eventually a typecast to int is performed.
    """
    if isinstance(value, torch.Tensor):
        if isinstance(space_obj, Box):
            v_min, v_max = space_obj.low, space_obj.high
        elif isinstance(space_obj, Discrete):
            v_min, v_max = 0, space_obj.n - 1
        elif isinstance(space_obj, tuple):
            v_min, v_max = space_obj
        else:
            raise NotImplementedError(type(space_obj))

        v_min = torch.tensor(v_min, device=value.device)
        v_max = torch.tensor(v_max, device=value.device)
        return torch.max(torch.min(value, v_max), v_min)

    if isinstance(space_obj, Box):
        return np.clip(value, a_min=space_obj.low, a_max=space_obj.high)

    elif isinstance(space_obj, Discrete):
        return np.maximum(np.minimum(np.round(value), space_obj.n - 1), 0).astype(int)

    raise NotImplementedError(type(space_obj))

