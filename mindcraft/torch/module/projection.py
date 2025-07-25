from torch.nn import Identity, Linear, Conv2d
from mindcraft.torch.util import get_conv2d_output_size
from numpy import array_equal, asarray, product, ndarray, stack
from numpy import arange as np_arange
from numpy import product as np_product
from typing import Union, Optional
from mindcraft.torch.activation import get_activation_function
from mindcraft.torch.module import Patchwork
from mindcraft.torch.util import layer_type_repr


class Projection(Patchwork):
    """ A simple identity projection layer for the `mindcraft` framework (might be useful for sensory neurons)

    (c) B. Hartl 2021
    """

    DEFAULT_LOCATE = 'mindcraft.torch.module.projection'

    REPR_FIELDS = ("input_size", "projection_size", "activation", *Patchwork.REPR_FIELDS)

    def __init__(self, input_size=None, projection_size=None, activation=None, **patchwork_kwargs):
        """ Constructs an `IdentityProjection` instance """
        self.network = None

        self.input_size = input_size
        self.projection_size = projection_size
        self.activation = activation
        Patchwork.__init__(self, **patchwork_kwargs)

    def _build(self):
        self._build_projection()
        self.activation = get_activation_function(self.activation)

    def _build_projection(self):
        self.network = Identity()

    def forward(self, x, *args):
        x = Patchwork.forward(self, x, *args)
        if self.activation is None:
            return self.network(x)
        return self.activation(self.network(x))

    def to(self, device):
        self.network = self.network.to(device)
        return Patchwork.to(self, device)

    def to_dict(self):
        dict_repr = Patchwork.to_dict(self)
        if dict_repr.get('activation', None):
            dict_repr['activation'] = layer_type_repr(dict_repr['activation'])
        return dict_repr


class LinearP(Projection):
    """ A simple linear projection layer for the `mindcraft` framework (may be useful for sensory neurons)

    (c) B. Hartl 2021
    """

    REPR_FIELDS = ("bias", *Projection.REPR_FIELDS)

    def __init__(self, input_size, projection_size=32, bias=True, **patchwork_kwargs):
        """ Constructs a Linear Projection instance

        :param input_size: the number of input channels (`shape[-1]`).
        :param projection_size: The projection size, i.e., the output-size of the module, defaults to 32.
        """
        self.bias = bias
        Projection.__init__(self, input_size=input_size, projection_size=projection_size, **patchwork_kwargs)

    def _build_projection(self):
        self.network = Linear(in_features=self.input_size, out_features=self.projection_size, bias=self.bias)


class PatchP(LinearP):
    """ A simple 2D-convolutional (patch) projection layer for the `mindcraft` framework
        (might be useful for sensory neurons)

    (c) B. Hartl 2021
    """

    REPR_FIELDS = ("kernel_size", "stride", "img_size", "padding", "padding_mode", "flatten",
                   *LinearP.REPR_FIELDS)

    def __init__(self, kernel_size=8, stride=None, input_size=3, projection_size=2, img_size=None,
                 padding=None, padding_mode='zeros', flatten=True, **patchwork_kwargs):
        self.kernel_size = kernel_size if hasattr(kernel_size, '__iter__') else [kernel_size, kernel_size]
        self.stride = stride or 1
        self.padding = padding or 0
        self.padding_mode = padding_mode
        self.flatten = flatten

        self.img_size = None
        self.shape_patches = None
        self.num_patches = None
        self.set_size(img_size)
        patchwork_kwargs["flatten_features"] = patchwork_kwargs.get("flatten_features", False)
        LinearP.__init__(self, input_size=input_size, projection_size=projection_size, **patchwork_kwargs)

    def _build_projection(self):
        self.network = Conv2d(self.input_size,
                              self.projection_size,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              padding_mode=self.padding_mode,
                              bias=self.bias,
                              dilation=1,
                              )

    def forward(self, x, *args):
        # set img_size property (and evaluate num_patches, shape_patches)
        x = self.resize(x)                         # shape: B, C, Nx, Ny
        x = LinearP.forward(self, x, *args)  #        B, Embd, Nx, Ny
        if self.flatten:
            x = x.flatten(-2)                          #        B, Embd, Nx times Ny
            x = x.transpose(1, 2)  # SEQUENCE, BATCH, FEATURES
        return x

    def set_size(self, img_size):
        if img_size is not None:
            if not hasattr(img_size, '__iter__'):
                img_size = [img_size, img_size]

            # kx = self.kernel_size if not hasattr(self.kernel_size, '__iter__') else self.kernel_size[0]
            # ky = self.kernel_size if not hasattr(self.kernel_size, '__iter__') else self.kernel_size[1]
            #
            # self.shape_patches = (img_size[0] // kx), (img_size[1] // ky)
            # self.num_patches = (img_size[0] // kx) * (img_size[1] // ky)

            self.shape_patches = get_conv2d_output_size(input_size=asarray(img_size),
                                                        kernel_size=asarray(self.kernel_size),
                                                        padding=self.padding,
                                                        stride=self.stride,
                                                        dilation=1,
                                                        )
            self.shape_patches = tuple([int(sp) for sp in self.shape_patches])
            self.num_patches = int(product(self.shape_patches))

            self.img_size = list(img_size)

        self.img_size = img_size

    def resize(self, x):
        if self.img_size is None:
            img_size = x.shape[-2:]   # assume shape (..., C, H=Nx, W=Ny)
            self.set_size(img_size=list(img_size))

        elif not array_equal(self.img_size, x.shape[-2:]):
            raise NotImplementedError(f"resize {x.shape[-2:]} to {self.img_size}")

        return x

    def get_coords(self, patches: Optional[Union[int, tuple, list, ndarray]] = None):
        unsqueeze = False
        if patches is None:
            return np_arange(self.num_patches)

        elif isinstance(patches, int):
            unsqueeze = True
            patches = [patches]

        patches = asarray(patches)
        patch_xy = stack([patches // self.shape_patches[1], patches % self.shape_patches[1]], axis=1)

        if unsqueeze:
            return patch_xy[0]

        return patch_xy

    # @property
    # def num_patches(self):
    #     from mindcraft.torch.module import Conv
    #
    #     kernel_size = self.kernel_size
    #
    #     img_size= self.img_size
    #     if not hasattr(img_size, "__iter__"):
    #         img_size = [img_size] * len(kernel_size)
    #
    #     stride = self.strid
    #     if not hasattr(stride, "__iter__"):
    #         stride = [stride] * len(kernel_size)
    #
    #     padding = self.padding
    #     if not hasattr(padding, "__iter__"):
    #         padding = [self.padding] * len(kernel_size)
    #
    #     p = [Conv.get_cnn_output_size(input_size=i, kernel_size=k, stride=s, padding=p)[0]
    #          for i, k, s, p in zip(img_size, kernel_size, stride, padding)
    #          ]
    #
    #     return int(np_product(p))
