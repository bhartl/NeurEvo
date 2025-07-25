import numpy as np
import h5py
import os
import ast


class Log:
    DEFAULT_LOG_FIELDS = ()  # ('episode', 'reward', 'done')

    def __init__(self, log_fields: (tuple, list) = DEFAULT_LOG_FIELDS, log_foos: (dict, None) = None):
        self.log_fields = None
        self._log_foos = None
        self.log_instructions = None
        self.log_history = None
        self.log_episode = 0
        self.set_log(log_fields=log_fields, log_foos=log_foos)

    def reset_log(self):
        self.log_history = []
        self.log_episode = 0

    def set_log(self, log_fields, log_foos=()):
        self.reset_log()

        self.log_fields = log_fields
        self.log_foos = log_foos if log_foos is not None else {}
        self.log_instructions = None
        self.set_log_instructions(log_foos)

    @property
    def log_foos(self):
        return self._log_foos

    @log_foos.setter
    def log_foos(self, value):
        self.set_log_instructions(value)
        self._log_foos = value

    def set_log_instructions(self, log_foos):
        self.log_instructions = dict()

        if log_foos not in ({}, (), [], None):
            for k, v in log_foos.items():
                if not hasattr(v, '__call__'):
                    self.log_instructions[k] = compile(v, "<string>", "eval")
                else:
                    self.log_instructions[k] = v

    def log(self, episode_key=None, no_attributes=False, **kwargs):
        if self.log_fields in ((), None):
            return None

        for k, v in kwargs.items():
            locals()[k] = v

        try:
            log_dict = self.log_history[self.log_episode]

        except IndexError:
            log_dict = {k: {} for k in self.log_fields}
            self.log_history.append(log_dict)

        values = []
        for k in self.log_fields:
            if k in self.log_instructions:
                foo = self.log_instructions[k]
                try:
                    v = foo() if hasattr(foo, '__call__') else eval(foo)
                except Exception as e:
                    raise ValueError(f'Error evaluating log instruction `{k}`: {e}')

            elif k in kwargs:
                v = kwargs[k]
            elif hasattr(self, k) and not no_attributes:
                v = getattr(self, k)
            else:
                values.append(None)
                continue
                # raise KeyError(f'can\'t find log field `{k}`')

            if episode_key:  #
                log_dict[k][episode_key] = v
            else:
                key, keys = 0, log_dict[k].keys()
                if len(keys) > 0:
                    key = max(keys) + 1

                log_dict[k][key] = v

            values.append(v)

        return values

    @classmethod
    def dump_history(cls, filename, log_history, exist_ok=False, key_offset=False):

        if not exist_ok:
            assert not os.path.isfile(filename), f"Specified file `{filename}` exists."

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if key_offset and isinstance(key_offset, bool):
            try:
                with h5py.File(filename, 'r') as h5:
                    keys = [int(k) for k in h5.keys()]
                    key_offset = max(keys) + 1

            except (OSError, ValueError):
                key_offset = 0

        with h5py.File(filename, 'a') as h5:
            dict_repr = Log.list_to_dict(log_history, offset=key_offset)
            Log.recursively_save_dict_contents_to_group(h5, '/', dict_repr)

        return key_offset

    @staticmethod
    def recursively_save_dict_contents_to_group(h5, path, dict_repr):
        """ Save items in `dict_repr` to the h5 file at the location specified by `path`. Nested dictionaries are
            recursively saved. """
        from torch import Tensor
        from mindcraft.torch.util import tensor_to_numpy
        for k, v in dict_repr.items():
            path_k = path + str(k)

            if isinstance(v, Tensor):
                v = tensor_to_numpy(v)

            if isinstance(v, np.ndarray) and v.ndim > 2:
                v = [vi for vi in v]

            if isinstance(v, (list, tuple)):
                v = Log.list_to_dict(v)

            if isinstance(v, (np.ndarray, np.int64, np.float64, str, bytes, int, float)) or v is None:

                if v is None:
                    v = np.nan

                try:
                    h5[path_k] = v
                except (OSError, RuntimeError):
                    if path_k[-1] == ".":
                        path_k = path_k[:-1] + "'.'"
                        h5[path_k] = v

            elif isinstance(v, dict):
                Log.recursively_save_dict_contents_to_group(h5, path_k + '/', v)
            else:
                raise ValueError(f'Do not understand type {type(v)}.')

    @staticmethod
    def list_to_dict(list_instance, offset=0):
        return {str(i + offset): vi for i, vi in enumerate(list_instance)}

    @staticmethod
    def dict_to_list(value_dict: dict, key_order=(), recursive=False):
        """ Returns list of value_dict values, ordered by their keys.

            :param value_dict: Dictionary, whose values are returned as list, according to the key_order parameter.
            :param key_order: iterable, defining the order of the values in value_dict in the returned list
                              of values.
                              If no key_order is specified, the keys of the value_dict are converted to a list and
                              sorted.
                              E.g., potential integer keys of a dictionary {0: 'element 0', 1: 'element 1', ...}
                              is converted to a list of ['element 0', 'element 1'], even if the keys are str
                              representations of integer values.
            :param recursive: Whether the dict should recursively be scanned for nested list-like dicts
            """

        if not key_order:

            key_order = []
            for k in value_dict.keys():

                if recursive and isinstance(value_dict[k], dict):
                    v = Log.dict_to_list(value_dict[k], recursive=True)
                    value_dict[k] = v

                try:
                    key_order.append(int(k))

                except (TypeError, ValueError):
                    key_order.append(k)

            key_order = np.asarray(key_order)
            try:
                assert not any(isinstance(k, str) for k in key_order)
                # assert np.array_equal(sorted(key_order), range(min(key_order), max(key_order)+1))
                return [list(value_dict.values())[k] for k in key_order.argsort()]

            except AssertionError:
                return value_dict

        return [value_dict[k] for k in key_order]

    @classmethod
    def load_history(cls, filename) -> list:
        with h5py.File(filename, 'r') as h5:
            loaded = cls.recursively_load_dict_contents_from_group(h5, '/')

            run_keys = loaded.keys()
            for k in run_keys:
                v = loaded[k]
                if "steps" in loaded[k]:
                    continue

                try:
                    steps = list(sorted([int(i) for i in list(v.values())[0].keys()]))
                    v["steps"] = steps

                except:
                    pass

            loaded = cls.dict_to_list(loaded, recursive=True)
            log_history = [cls.wrap_np(li) for li in loaded]
            return log_history

    @staticmethod
    def recursively_load_dict_contents_from_group(h5, path='/') -> dict:
        """
        ....
        """
        ans = {}
        for key, item in h5[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[()]  # access item.value
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = Log.recursively_load_dict_contents_from_group(h5, path + key + '/')

        return ans

    @staticmethod
    def wrap_np(log_dict):
        for k, v in log_dict.items():
            try:
                try:
                    steps = sorted(v.keys())
                    log_dict[k] = np.array([v[s] for s in steps])

                except AttributeError:
                    log_dict[k] = np.array(v)

            except ValueError:
                if isinstance(v, list) and all(isinstance(vi, np.ndarray) for vi in v):
                    log_dict[k] = Log.wrap_np({str(i): vi for i, vi in enumerate(v)})

        return log_dict


def crawl_for_log_files(path: str, rel_path=False):
    """ Crawls through the specified path and returns a list of all log files. """
    log_files = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".log"):
                file_path = os.path.join(root, file)
                if rel_path:
                    log_files.append(os.path.relpath(file_path, path))
                else:
                    log_files.append(file_path)

    return log_files


def key_value_pairs(path, postfix=""):
    """ Extract ((key, val), ...)-pairs from a `path` of the format
        `"<some prefix>/key1_val1-key2_val2/somekey_3-abc_d-<some postfix>"`

    Example
    >>> key_value_pairs("<some prefix>/key1_val1-key2_val2/somekey_3-abc_d-<some postfix>")
    (('key1', 'val1'), ('key2', 'val2'), ('somekey', 3), ('abc', 'd'))
    """
    if postfix:
        path = path.replace(postfix, "")

    path = path.replace("-", "/")  # .replace(".", "/")
    key_val_pairs = [p.split("_") for p in path.split('/') if len(p.split("_")) == 2]

    for i in range(len(key_val_pairs)):
        try:
            key_val_pairs[i][1] = ast.literal_eval(key_val_pairs[i][1])
        except:
            pass

    return tuple([tuple(kv) for kv in key_val_pairs])


def crawl_logfiles_dataframe(directory, verbose=False, postfix=""):
    """ Crawls through the specified directory and returns a pandas.DataFrame with the key-value pairs
        extracted from the log file names.

        The log file names are expected to be of the format
          - `"<some prefix>/key1_val1-key2_val2/somekey_3-abc_d-<some postfix>.log"`
        resulting in the key-value pairs
          - `(key1, val1), (key2, val2), (somekey, 3), (abc, d)`W
    """
    if verbose:
        print(f"crawl '{directory}'")

    logfiles = [f for f in crawl_for_log_files(directory) if not postfix or f.endswith(postfix)]
    if verbose:
        [print("-", log) for log in logfiles]

    import pandas as pd
    df = pd.DataFrame([{k: v for k, v in key_value_pairs(file, postfix=postfix)} for file in logfiles])
    df["logfile"] = logfiles

    checkpoints = [logfile.replace(".log", "-ckpt.h5") for logfile in logfiles]
    df["checkpoint"] = checkpoints

    return df
