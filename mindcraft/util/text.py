from typing import Union
import ast


def fill_pattern(pattern: str, wildcards: Union[dict, str]):
    """Fills all `key` appearances in a string `pattern` with all combinations of `[values]` of a `wildcards`
    mapping {`key`: [values]}.

    Example

    >>> fill_pattern(pattern = "test_{T}-num_{N}", wildcards = {"{T}": ["u", "v"], "{N}": [4, 2], "{NOT A KEY}": 1})
    ['test_u-num_4', 'test_u-num_2', 'test_v-num_4', 'test_v-num_2']
    """
    if isinstance(wildcards, str):
        import json
        wildcards = json.loads(wildcards)

    if not wildcards:
        return [pattern]

    k = list(wildcards.keys())[0]
    v = wildcards[k]
    if not hasattr(v, '__iter__') or isinstance(v, str):
        v = [v]

    if not len(v):
        return [pattern]

    other_wildcards = {kj: vj for kj, vj in wildcards.items() if kj != k}

    filled = []
    for vi in v:
        filled_vi = fill_pattern(pattern.replace(k, str(vi)), wildcards=other_wildcards)
        for fi in filled_vi:
            if fi not in filled:
                filled.append(fi)

    return filled


def key_value_pairs(path):
    """ Extract ((key, val), ...)-pairs from a `path` of the format
        `"<some prefix>/key1_val1-key2_val2/somekey_3-abc_d-<some postfix>"`

    Example
    >>> key_value_pairs("<some prefix>/key1_val1-key2_val2/somekey_3-abc_d-<some postfix>")
    (('key1', 'val1'), ('key2', 'val2'), ('somekey', 3), ('abc', 'd'))
    """
    path = path.replace("-", "/").replace(".", "/")
    key_val_pairs = [p.split("_") for p in path.split('/') if len(p.split("_")) == 2]
    for i in range(len(key_val_pairs)):
        try:
            key_val_pairs[i][1] = ast.literal_eval(key_val_pairs[i][1])
        except:
            pass

    return tuple([tuple(kv) for kv in key_val_pairs])
