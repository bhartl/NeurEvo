from pydoc import locate
from copy import deepcopy
from inspect import getmembers, isclass
from mindcraft.util.types import get_default_args, locate_type_of
import os
import re

# possible useful globals for make
import numpy as np
from numpy import *
from yaml import safe_dump as yml_safe_dump


class Repr:
    REPR_FIELDS = ()
    DEFAULT_LOCATE = None

    def __init__(self, repr_fields=(), omit_default=False, to_list=()):
        self.repr_fields = repr_fields or deepcopy(self.REPR_FIELDS)
        self.omit_default = omit_default
        self.to_list = to_list

    def to_dict(self):
        loc, cls = locate_type_of(self)
        dict_repr = dict(locate=loc, cls=cls.__name__)
        # default_signature = get_default_args(self) if self.omit_default else {}
        for f in self.repr_fields:
            attr = getattr(self, f)
            # default = default_signature.get(f, None)  # NOTE: recursion-depth error due to nested to_dict call
            if attr is None:                            # -> should be `attr is default`
                continue  # omit by "id"

            # elif not hasattr(attr, '__iter__') and not hasattr(default, '__iter__') and attr == default:
            #     continue  # omit by "scalar equality"
            #
            # elif (hasattr(attr, '__iter__') or hasattr(default, '__iter__')) and array_equal(attr, default):
            #     continue  # omit by array equality

            if hasattr(attr, "to_dict"):
                attr = attr.to_dict()

            elif hasattr(attr, "todict"):
                attr = attr.todict()

            if f in self.to_list:
                attr = self.v_to_list(attr)

            dict_repr[f] = attr

        return dict_repr

    @classmethod
    def v_to_list(cls, v):
        if hasattr(v, "to_list"):
            return v.to_list()

        elif hasattr(v, "tolist"):
            return v.tolist()

        else:
            return list(v)

    @classmethod
    def from_dict(cls, dict_repr):
        # dict_repr = deepcopy(dict_repr)

        class_ = dict_repr.get('cls', cls)
        locate_ = dict_repr.get('locate',  cls.DEFAULT_LOCATE)
        dict_repr = {k: v for k, v in dict_repr.items() if k not in ('cls', 'locate')}

        if isinstance(class_, str):
            if locate_ is not None:
                cls_ = locate('.'.join([locate_, class_]))
            else:
                cls_ = locate(class_)
        else:
            cls_ = class_

        assert isinstance(cls_, type), f'{repr(cls.__name__)}.from_dict: ' \
                                       f'type `{type(cls_)}` of repr ' \
                                       f'{(repr(locate_) + ".") if locate_ is not None else ""}' \
                                       f'{repr(class_)} not understood.'
        try:
            return cls_(**dict_repr)

        except Exception as ex:
            if len(ex.args) > 0:
                args = (a for i, a in enumerate(ex.args) if i > 0)
                ex.args = (f'{cls}.{ex.args[0]}', *args)

            raise

    def to_json(self, filename=None, indent=2, **kwargs):
        import json
        dict_repr = self.to_dict()

        if filename is not None:
            if os.path.dirname(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as s:
                return json.dump(dict_repr, fp=s, indent=indent, **kwargs)
        else:
            return json.dumps(dict_repr, indent=indent, **kwargs)

    def to_yml(self, filename=None, default_flow_style=False, **kwargs):
        dict_repr = self.to_dict()

        if filename is not None:
            if os.path.dirname(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as s:
                return yml_safe_dump(dict_repr, stream=s, default_flow_style=default_flow_style, sort_keys=False, **kwargs)
        else:
            # yaml.safe_dump(dict_repr, round, default_flow_style=default_flow_style, sort_keys=False, **kwargs)
            return yml_safe_dump(dict_repr, default_flow_style=default_flow_style, sort_keys=False, **kwargs)

    def __repr__(self):
        dict_repr = self.to_dict()
        cls = dict_repr.pop('cls')
        _ = dict_repr.pop('locate', None)
        kwargs = [f'{k}={repr(v)}' for k, v in dict_repr.items()]
        return f"{cls}({', '.join(kwargs)})"

    @classmethod
    def get_partial_local(cls, **others):
        return {cls.__name__: cls, **others}

    @classmethod
    def make(cls, repr_obj, partial_local=None, **repr_kwargs):
        """
        :param repr_obj: representable object of class `cls`, could be repr string, dict or instance of cls
        :param partial_local: lookup mapping added to local for repr evaluation
        :param repr_kwargs: keyword value pairs overwriting repr_obj's fields in the initialized instance (for dynamic purposes)
        :returns: `cls` instance
        """
        if repr_obj is None:
            return None

        if partial_local is None:
            partial_local = {}

        if isinstance(repr_obj, cls):
            for k, v in repr_kwargs.items():
                setattr(repr_obj, k, v)

            return repr_obj

        for k, v in partial_local.items():
            if isinstance(v, str):
                # try loading string repr of necessary classes for eval
                partial_v = locate(v)
                assert partial_v is not None, f"{v}"
                locals()[k] = partial_v

            else:
                locals()[k] = v

        if isinstance(repr_obj, dict):
            # try loading from dictionary representation
            for k, v in repr_kwargs.items():
                repr_obj[k] = v

            return cls.from_dict(repr_obj)

        try:
            try:
                if cls.__name__ not in locals():
                    locals()[cls.__name__] = cls

                eval_obj = eval(repr_obj)
                assert isinstance(eval_obj, cls), f"cls: `{cls.__name__}`, repr_obj: `{repr_obj}`"

                for k, v in repr_kwargs.items():
                    setattr(eval_obj, k, v)

                return cls.make(repr_obj=eval_obj, partial_local=partial_local, **repr_kwargs)

            except NameError:
                from mindcraft.io import spaces
                partial_make = {name: class_ for name, class_ in getmembers(spaces, isclass)}
                assert not all([k in partial_local for k in partial_make.keys()])
                return cls.make(repr_obj, partial_local={**partial_local, **partial_make}, **repr_kwargs)

        except (AssertionError, SyntaxError):
            if isinstance(repr_obj, str) and cls.is_valid_path(repr_obj):
                if not os.path.isfile(repr_obj):
                    raise FileNotFoundError(f"Could not find file at {repr_obj} from working directory {os.getcwd()}.")

                # try loading from file representation (yaml, json or plain text repr)
                with open(repr_obj, 'r') as f:
                    if repr_obj.endswith('.yml') or repr_obj.endswith('.yaml'):
                        from yaml import safe_load
                        loaded_repr = safe_load(f)

                    elif repr_obj.endswith('.json'):
                        from json import load
                        loaded_repr = load(f)

                    else:
                        loaded_repr = ''.join(f.readlines()).strip()

                return cls.make(loaded_repr, partial_local=partial_local, **repr_kwargs)

    @classmethod
    def is_valid_path(cls, potential_path: str):
        if not re.search(r'[^A-Za-z0-9_\-\\/öäüÖÄÜ@]', potential_path):
            return False

        return True

    @classmethod
    def recursively_override(cls, dst: dict, src, key: str = None):
        """ Recursively override attributes in the `dst` dictionary with
            content from `src`:

            all (sub)dict content of `src` will be aligned with `dst` (branches),
            and all non-dict content (leaves) will be overwritten.

        :param dst: The destination dictionary whose content is going to be overwritten by src content.
        :param src: The source dictionary whose content will override `dst` content.
        :param key: An optional key used to propagate recursively into the dst / src dictionaries.
        """
        if key is None:
            for k, v in src.items():
                cls.recursively_override(dst, v, k)
        elif not isinstance(src, dict):
            dst[key] = src
        elif key not in dst:
            dst[key] = src
        elif not isinstance(dst[key], dict):
            dst[key] = src
        else:
            for k, v in src.items():
                cls.recursively_override(dst[key], v, k)
