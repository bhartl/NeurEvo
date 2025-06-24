import torch
from torch.nn import Module
from mindcraft.torch.module import Patchwork


class Ensemble(Patchwork):
    REPR_FIELDS = ("nn", "redundancy", "randomize", "hooks", "foo", *Patchwork.REPR_FIELDS)

    def __init__(self, nn: Patchwork, redundancy: int = 1, randomize: bool = False, hooks=(), foo="mean", 
                 **patchwork_kwargs):
        Module.__init__(self)
        
        nn = Patchwork.to_dict(Patchwork.make(nn))
        self.nn = {k: v for k, v in nn.items() if k not in ['recover_indices', 'serialized', 'serialize_mask']}        
        self.redundancy = redundancy
        self.randomize = randomize
        self.hooks = hooks
        self.foo = foo
        self.nn_stack = []
        self.nn_labels = []
        Patchwork.__init__(self, **patchwork_kwargs, to_list=["hooks", ])

    def _build(self):
        self.nn_stack = [Patchwork.make(self.nn) for _ in range(self.redundancy)]
        for i, nn_i in enumerate(self.nn_stack):
            key_i = f"nn_{i}"
            setattr(self, key_i, nn_i)
            self.nn_labels.append(key_i)
        
    def to_dict(self):
        dict_repr = Patchwork.to_dict(self)
        return dict_repr

    def forward(self, x, *args):
        self.apply_hooks()
        y = torch.stack([nn.forward(x, *args) for nn in self.nn_stack])
        self.apply_hooks()
        return self.aggregate(y)

    def apply_hooks(self):
        for hook in self.hooks:
            if hook == "states":
                continue

            hook_value = self.get_hook(hook)
            try:
                hook_aggregate = self.aggregate(hook_value)
                self.set_hook(hook, hook_aggregate)
            except TypeError:
                continue

    def get_hook(self, key):
        hook_values = [getattr(patchwork, key) for patchwork in self.nn_stack]
        if all(isinstance(h, torch.Tensor) for h in hook_values):
            return

        if all(isinstance(h, tuple) for h in hook_values):
            return tuple(torch.stack(list(x), dim=0) for x in zip(*hook_values))

        return hook_values

    def aggregate(self, values):
        foo = getattr(torch, self.foo)
        if isinstance(values, tuple):
            return tuple(foo(v, dim=0) for v in values)
        return foo(values, dim=0)

    def set_hook(self, key, value):
        for patchwork, hook_value in zip(self.nn_stack, value):
            setattr(patchwork, key, hook_value)

    @property
    def input_size(self):
        return self.nn_stack[0].input_size

    @property
    def output_size(self):
        return self.nn_stack[0].output_size

    @property
    def is_sequence_module(self):
        return self.nn_stack[0].is_sequence_module

    @property
    def states(self):
        if self.is_sequence_module and self.nn_stack[0].states is not None:
            states = self.get_hook("states")
            if "states" in self.hooks:
                states = self.aggregate(states)
            else:
                states = tuple(torch.concatenate([si for si in s], dim=0) for s in states)
            return states
        return None

    @states.setter
    def states(self, value):
        if self.is_sequence_module:
            for i, nn in enumerate(self.nn_stack):
                if "states" in self.hooks:
                    nn.states = list(v.clone() for v in value)
                else:
                    nn.states = list(v[i * nn.num_layers:(i+1) * nn.num_layers] for v in value)

    @property
    def num_layers(self):
        if self.is_sequence_module:
            num_layers = self.nn_stack[0].num_layers
            if "states" in self.hooks:
                return num_layers
            return num_layers * self.redundancy
        return 0
