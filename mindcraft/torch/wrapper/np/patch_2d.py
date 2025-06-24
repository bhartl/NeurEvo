from numpy import empty, ndarray
from itertools import product
from typing import Union


class Patch2D(object):
    """ A tensor transformation layer which transform a 2D input tensor (with optional leading batch and channel
        dimensions) into a 2D tensor of patches of dimension `kernel_size`.

        Overlapping patches can be generated using the `stride` parameter.

        The transformation looks as follows: an input tensor of shape
        `(..., DIM-X, DIM-Y)` -> `(..., NX, NY, KERNEL_SIZE-X, KERNEL_SIZE-Y)`,
        where `NX` and `NY` are functions of `DIM-X`, `DIM-Y`, `kernel_size` and `stride`.

        A mapping of the `NX x NY` patches to their position in the original `DIM-X x DIM-Y` input tensor
        can be retrieved via the `get_patch_positions()` method.

        The layer can be equipped with a positional encoding (which is either None, learnable or fixed).
        And the positional encoding mode can be specified (None, add, concatenate)

        see https://attentionagent.github.io/

        (c) B. Hartl 2021
        """

    POSITIONAL_ENCODING_LEARNABLE = 'learnable'
    
    POSITIONAL_ENCODING_CAT_MODE = "cat"
    POSITIONAL_ENCODING_ADD_MODE = "add"

    def __init__(self,
                 kernel_size: Union[int, tuple],
                 stride: Union[int, tuple] = None,
                 channels_first=True,
                 patch_constant: bool = True,
                 ):
        """ Constructs a Patch2D Layer

        :param kernel_size: int or tuple specifying the kernel size in x and y directions.
        :param stride: (Optional) int or tuple specifying the strides in x and y directions (defaults to None,
                       i.e., `kernel_size` is used as stride for non-overlapping patches).
        :param channels_first: Boolean specifying whether channels before or after the xy-dimensions (defaults to True)
        :param patch_constant: Boolean specifying whether the patch-size and image size is constant throughout
                               evaluation (defaults to True)
        """

        self._kernel_size = None
        self.kernel_size = kernel_size

        self._stride = None
        self.stride = stride

        self.channels_first = channels_first

        # helpers
        self.patch_constant = patch_constant
        self.patch_xy = None
        self.patch_position = None
        
        self._patch_shape = None
        self._xy_shape = None

    @property
    def kernel_size(self) -> tuple:
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value: Union[int, tuple]):
        if hasattr(value, '__iter__'):
            self._kernel_size = tuple(value)
        else:
            self._kernel_size = (value, value)

        assert len(list(self._kernel_size)) == 2

    @property
    def stride(self) -> tuple:
        return self._stride

    @stride.setter
    def stride(self, value: Union[int, tuple, None]):
        if value is None:
            value = self.kernel_size

        if hasattr(value, '__iter__'):
            self._stride = tuple(value)
        else:
            self._stride = (value, value)

        assert len(list(self._stride)) == 2

    def get_patch_xy_shape(self, patch: Union[None, object] = None) -> tuple:
        if self.patch_xy is not None and not self.patch_constant:
            return self.patch_xy

        if patch is None:
            shape = self._patch_shape
        else:
            shape = patch.shape

        if not shape:
            return None, None

        idx_offset = 4 + (not self.channels_first)
        x_dim, y_dim = 0 + len(shape) - idx_offset, 1 + len(shape) - idx_offset

        return shape[x_dim], shape[y_dim]

    def get_patch_positions(self, patch: Union[None, object] = None) -> ndarray:
        """ Returns a tensor of the same (xy) shape as the `patch` argument (or the input after a call of the layer
            if patch is None), which specifies the (x, y) positions of each patch in the original 2D tensor.

        :param patch: Optional patch_2d result (defaults to None -> layer needs to be called first)
        :returns: a 3D tensor where dim_0 and dim_1 correspond to the patch indices and dim_2 specifies the
                  xy-coordinates of each patch in the original 2D input.
        """

        if self.patch_position is not None and not self.patch_constant:
            return self.patch_position

        x_shape, y_shape = self.get_patch_xy_shape()

        patch_pos = empty(shape=(x_shape, y_shape, 2))
        for dim, (kernel, stride) in enumerate(zip(self.kernel_size, self.stride)):
            for batch_xy in product(range(x_shape), range(y_shape)):
                pos_xy = batch_xy[dim] + batch_xy[dim] * int(kernel * (stride // kernel) + stride % kernel - 1)
                patch_pos[batch_xy[0], batch_xy[1], dim] = pos_xy

        self.patch_position = patch_pos
        return patch_pos

    def forward(self, x: object) -> object:
        """ applies the layer to transform a 2D input tensor (with optional leading batch and channel dimensions)
        into a 2D tensor of patches of dimension `kernel_size`. Overlapping patches can be generated using the
        `stride` parameter.

        Note: A mapping of the `NX x NY` patches to their position in the original `DIM-X x DIM-Y` input tensor
        can be retrieved via the `get_patch_positions()` method.

        :param x: Input tensor of shape `(..., DIM-X, DIM-Y)`
        :returns: Patches of the input tensor of shape `(..., NX, NY, KERNEL_SIZE-X, KERNEL_SIZE-Y)`,
                  where `NX` and `NY` are functions of `DIM-X`, `DIM-Y`, `kernel_size` and `stride`.
        """

        raise NotImplementedError
