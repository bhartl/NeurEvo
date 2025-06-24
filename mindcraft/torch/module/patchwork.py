from torch import Tensor, tensor, cat, load, save
from torch import float as torch_float
from torch.nn import Module
from numpy import ndarray
from mindcraft.io import Repr
from mindcraft.torch.util import flatten_parameters
from mindcraft.torch.util import tensor_to_numpy
from mindcraft.torch.util import recover_flattened
from mindcraft.torch.util import set_parameters
from abc import ABC
from typing import Optional, Union
from collections import OrderedDict


class Patchwork(Module, Repr, ABC):
    """  PyTorch Module Wrapper of the `mindcraft` Framework

    - Allows serializing all `torch` parameters
    - Can be (re)stored (from)as `yml`-file

    (c) B. Hartl 2021
    """
    REPR_FIELDS = ('retain_grad', 'checkpoint_path', 'flatten_features',
                   'recover_indices', 'serialize_mask', 'is_nested',
                   # 'serialized'  # needs to be omitted, no property - manually set in Patchwork.to_dict
                   )

    DEFAULT_LOCATE = 'mindcraft.torch.module'

    def __init__(self,
                 retain_grad: bool = True,
                 checkpoint_path: Optional[str] = None,
                 flatten_features: bool = False,
                 recover_indices: tuple = None,
                 serialized: Optional[Union[Tensor, ndarray]] = None,
                 serialize_mask: dict = None,
                 device: str = 'cpu',
                 is_nested: bool = False,
                 **kwargs
                 ):
        """

        :param retain_grad:
        :param checkpoint_path:
        :param flatten_features:
        :param recover_indices:
        :param serialized:
        :param serialize_mask:
        :param device:
        :param is_nested: Boolean flag to mark a model as is_nested patchwork model. Serializable information is not included
                          in dict representations - it is assumed at higher levels.
        :param kwargs:
        """
        if not hasattr(self, '_modules'):
            Module.__init__(self)

        kwargs['omit_default'] = kwargs.get('omit_default', True)
        kwargs['to_list'] = list(kwargs.get('to_list', [])) + ['serialize_mask']
        Repr.__init__(self, repr_fields=self.REPR_FIELDS, **kwargs)

        self._recover_indices = None
        self.recover_indices = recover_indices
        self.serialize_mask = serialize_mask

        self._build()

        self.checkpoint_path = checkpoint_path
        if serialized is not None:
            self.deserialize_parameters(serialized)

        elif checkpoint_path is not None:
            self.load_parameters()

        self._retain_grad = None
        self.retain_grad = retain_grad
        self.flatten_features = flatten_features
        self.is_nested = is_nested
        self._device = device
        self.to(device)

    def to(self, device):
        Module.to(self, device)
        self._device = device
        return self

    def copy(self):
        return type(self).make(self.to_dict())

    def clone(self):
        return self.copy()

    @classmethod
    def load_checkpoint(cls, checkpoint_path) -> Union['Patchwork', dict, OrderedDict, None]:
        if checkpoint_path is None:
            return None

        p: Patchwork = load(checkpoint_path)
        return p

    def load_parameters(self, checkpoint_path=None) -> Union[None, 'Patchwork', dict, OrderedDict]:
        checkpoint_path = checkpoint_path or self.checkpoint_path
        p: Union[None, dict, OrderedDict, Patchwork] = self.load_checkpoint(checkpoint_path)

        if p is not None:
            if isinstance(p, (dict, OrderedDict)):
                set_parameters(state_dict=p, module=self)

            elif isinstance(p, Patchwork):
                self.deserialize_parameters(p.serialize_parameters())
                self.checkpoint_path = checkpoint_path

            elif isinstance(p, (ndarray, tuple)):
                self.deserialize_parameters(p)
                self.checkpoint_path = checkpoint_path

            else:
                raise TypeError(f"loaded checkpoint `{p}` not understood.")

        return p

    def save_parameters(self, file_name):
        save(self.state_dict(), file_name)

    @property
    def recover_indices(self):
        if self._recover_indices is None:
            _, self._recover_indices = flatten_parameters(self, calc_indices=True, mask=self.serialize_mask)

        return self._recover_indices

    @recover_indices.setter
    def recover_indices(self, value):
        self._recover_indices = value

    def serialize_parameters(self, to_numpy=True):
        flattened_params = flatten_parameters(self, calc_indices=False, mask=self.serialize_mask)

        if to_numpy:
            return tensor_to_numpy(flattened_params)

        return flattened_params

    def deserialize_parameters(self, serialized):
        if serialized is None:
            return self.parameters()

        if not isinstance(serialized, Tensor) and len(serialized):
            serialized = Tensor(serialized)

        return recover_flattened(flat_parameters=serialized,
                                 indices=self.recover_indices,
                                 model=self,
                                 update_model=True,
                                 mask=self.serialize_mask)

    def to_dict(self):
        dict_repr = Repr.to_dict(self)
        if self.is_nested:
            dict_repr.pop('recover_indices', None)
            dict_repr.pop('serialized', None)
            dict_repr.pop('serialize_mask', None)
        else:
            dict_repr['recover_indices'] = [[t[0], [int(idx) for idx in t[1]]] for t in self.recover_indices]
            dict_repr['serialized'] = self.serialize_parameters(to_numpy=True).tolist()

        return dict_repr

    @property
    def retain_grad(self):
        return self._retain_grad

    @retain_grad.setter
    def retain_grad(self, value):
        for parameter in self.parameters():
            parameter.requires_grad = value

        if not value:
            self.eval()

        self._retain_grad = value

    def __repr__(self):
        if self.checkpoint_path is not None:
            return f"{self.__class__.__name__}.make({repr(self.checkpoint_path)})"

        return f"{self.__class__.__name__}.make({repr(self.to_dict())})"

    def forward(self, x, *args: Tensor) -> Tensor:
        """ Merge flattened features, but keep batch-size and seq-len.

        :param x: variable numbers of input feature-tensor
        """
        retain_grad = any(xi.requires_grad for xi in x if xi is not None)
        if args:  # merge into last dimension
            x = cat([xi.view(*(xi.shape[i] for i in range(min(2, len(xi.shape) - 1))), -1)
                     for xi in (x, *args) if xi is not None], dim=-1)

        if len(x.shape) > 3 and self.flatten_features:  # flatten features -> [BATCH, SEQ-LEN, FEAT]
            x = x.view(x.shape[0], x.shape[1], -1)

            if retain_grad:
                x.retain_grad()

        return x

    @property
    def is_sequence_module(self):
        return False

    def _build(self):
        raise NotImplementedError("`_build` method called before `deserialize_parameters`.")

    @property
    def device(self):
        return self._device

    def to_tensor(self, *args):
        return (a if isinstance(a, Tensor) else tensor(a, device=self.device, dtype=torch_float,
                                                       requires_grad=self.retain_grad) for a in args)

    def aggregates(self, dim):
        return False

    @property
    def parameters_str(self):
        p_str = ""
        max_len = 0
        num_params = 0
        num_trainables = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                p_line = f"{name}\t{tuple(param.data.shape)}\n"
                p_str += p_line
                max_len = max([max_len, len(p_line)])
                num_trainables += len(param.flatten())
            num_params += len(param.flatten())

        name = self.__class__.__name__ + "\n"
        max_len = max([max_len, len(name)])

        sep = "=" * max_len + "\n"
        p_str = sep + name + sep + p_str + sep
        p_str += f"num-params    : {num_params}\n"
        p_str += f"num-trainables: {num_trainables}\n"
        p_str += sep
        return p_str

    # @classmethod
    # def make(cls, repr_obj, complexify=False, complexify_kwargs=None, **make_kwargs):
    #     patchwork = cls.make(repr_obj, **make_kwargs)
    #     if complexify:
    #         from mindcraft.torch.module import Complexify
    #         if not isinstance(patchwork, Complexify):
    #             if complexify_kwargs is None:
    #                 complexify_kwargs = {}
    #             patchwork = Complexify(real=patchwork, **complexify_kwargs)
    #
    #     return patchwork
