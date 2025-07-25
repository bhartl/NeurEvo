import yaml
import os
import logging as log
import copy
import json
from mindcraft.util.text import fill_pattern
from numpy import argmax


def get_wildcard_mask(wildcard_mask, prefix="", postfix=""):
    config_mask = "-".join([f"{name}_{wildcard}" for name, wildcard in wildcard_mask])
    return prefix + config_mask + postfix


def get_scan_wildcard_mask(wildcards, config_prefix="", agent_prefix="", postfix=""):
    if config_prefix:
        if os.path.isdir(config_prefix) and not config_prefix.endswith("/"):
            config_prefix += "/"
    config_mask = get_wildcard_mask(wildcards["config"], prefix=config_prefix, postfix=postfix)

    if agent_prefix:
        if os.path.isdir(agent_prefix) and not agent_prefix.endswith("/"):
            agent_prefix += "/"
    agent_mask = get_wildcard_mask(wildcards["agent"], prefix=agent_prefix, postfix=postfix)

    return config_mask, agent_mask


def copy_to(dst, key, val):
    for ki in key.split(".")[:-1]:
        dst = dst[ki]

    dst[key.split(".")[-1]] = val


def get_val(src, key):
    for ki in key.split("."):
        src = src[ki]
    return src


def replace_key_val(source, destination, self_ref, cross_ref, references=()):
    # source-format:
    #   list of dicts of the form
    #     { key: ...,
    #       val: [...],               # list of values to set `key` in `config_mask`
    #       alias: [...],             # corresponding aliases of `val` for wildcard
    #       wildcard: [key, wildcard],  # wildcard
    #       ref: {...}                # optional referencing for relational values, see below
    #     }
    refined_tasks = []
    refined_references = []
    for i, (value, alias) in enumerate(zip(source["val"], source.get("alias", source["val"]))):
        log.info(f"- mapping source-key '{source['key']}' to value `{value}`")

        for t in destination:
            refined_task = copy.deepcopy(t)
            refined_task["wildcard"].append(list(source["wildcard"]) + [alias])  # [name, wildcard, value]
            copy_to(dst=refined_task["config"], key=source["key"], val=value)

            refs = source.get("ref", None)
            if refs:
                # branch-format:
                #   dict of the form
                #   { [Optional] config: [...],
                #     [Optional] agent: [...],
                #   }
                crefs = refs.get(self_ref, None)
                arefs = refs.get(cross_ref, None)

                # refs specifies a dictionary of the form
                #   { key: ...,
                #     val: [...],
                #     [Optional for agent] product: product_key,
                #     [Optional for agent] pattern: pattern_str,
                #   }
                #   where `key` corresponds to related entries either in `config` or `agent` that are
                #   filled by the corresponding `ref-val`ues (must be of same length as [val] above).
                if crefs:  # todo: consider config reference from agent scan setting
                    for cref in crefs:
                        k, v = cref["key"], cref["val"][i]
                        log.info(f"  mapping config-ref '{k}' to value `{v}`")
                        copy_to(dst=refined_task["config"], key=k, val=v)

                #   Optionally for agents, a string pattern can be specified that are filled by the product of
                #   `val x product_key-val`. This could be necessary for functionally related entries.
                if arefs:
                    for agent in references:  # todo: consider agent reference from agent scan setting
                        refined_ref = copy.deepcopy(agent)
                        for aref in arefs:
                            k, v = aref["key"], aref["val"][i]
                            pattern = aref.get("pattern", None)  # only apply ref-val if "pattern" present in agent
                            if pattern:
                                try:
                                    if pattern != get_val(refined_ref['config'], key=k):
                                        # pattern does not occur in refined_agent, skip
                                        continue
                                except KeyError:
                                    continue

                                if "product" in aref:
                                    v *= get_val(refined_ref['config'], key=aref["product"])

                                elif "sum" in aref:
                                    v += get_val(refined_ref['config'], key=aref["sum"])

                            log.info(
                                f"  mapping agent-ref '{k}' to value `{v}` for agent '{refined_ref['filename']}'")
                            copy_to(refined_ref['config'], key=k, val=v)
                        refined_ref['wildcard'].append(
                            list(source["wildcard"]) + [alias])  # [name, wildcard, value]
                        refined_references.append(refined_ref)

            refined_tasks.append(refined_task)

    if not references:
        return refined_tasks

    return refined_tasks, refined_references or references


def dump_scan(scan_config, config_source, agent_sources, destination="data/scan/", dev=False,
              config_dst="config/", agent_dst="agent/"):
    """Applies n2n mapping of properties listed in a `scan_config` yml-file to the specified keys
       in a `config_source` yml-file and to a collection of `agent_sources` (a folder containing
       yml files, or a dict of {agent-name: agent-content}) and writes the respectively adapted configuration
       file and the corresponding agent into a `destination/config` and `destination/agent` directory.

    :param scan_config: Path to scan configuration file. See inline comments below and checkout example file
                        `mindcraft/tests/mindcraft_test/util/hpc/scan_instructions.yml` and comments therein.
    :param config_source: Path to source configuration file that is modified by the instructions in `scan_config`;
                          all variations of modifications are exported to `<destination>/config/*.yml`.
    :param agent_sources: Path to folder containing agent-yml files that are to be modified by the instructions in
                          `scan_config` in accordance with the `config_source`;
                          all variations of modifications are exported to `<destination>/agent/*.yml`.
    :param destination: Destination folder for all modifications of the `config_source` and agents in `agent_sources`,
                        defaults to "data/scan/"-
    :param dev: Boolean flag to map a `dev: [{'key': ..., 'val': ...}]` list in `scan_config` to key-val pairs in
                `config_source`; this is meant to map dev-settings globally to the modified configurations and agents.
    :param config_dst: Destination folder for all modified configurations, defaults to "config/".
    :param agent_dst: Destination folder for all modified agents, defaults to "agent/".
    """

    os.makedirs(destination, exist_ok=True)
    log.basicConfig(
        level=log.INFO,
        format="%(asctime)s > %(message)s",
        handlers=[
            log.FileHandler(os.path.join(destination, "scan.log")),
            log.StreamHandler()
        ]
    )

    log.info(f"Loading `scan_config` '{scan_config}'")
    with open(scan_config, "r") as s:
        scan_cmd = yaml.safe_load(s)

    log.info(f"Loading `config_source` '{config_source}'")
    with open(config_source, "r") as s:
        config_mask = yaml.safe_load(s)

    if isinstance(agent_sources, str):
        log.info(f"Loading agents from '{agent_sources}'")
        agent_masks = {}
        for agent_file in os.listdir(agent_sources):
            if agent_file.endswith(".yml"):
                log.info(f"- found file '{agent_file}'")
                with open(os.path.join(agent_sources, agent_file), "r") as a:
                    agent_masks[agent_file] = yaml.safe_load(a)
    else:
        agent_masks = agent_sources

    if dev:
        # dev-format:
        #   list of dicts of the form
        #     { key: ...,
        #       val: ...,
        #     }
        log.info(f"Preparing `dev` settings:")
        scan_dev = {d["key"]: d["val"] for d in scan_cmd["dev"]}
        for k, v in scan_dev.items():
            log.info(f"- mapping def-key '{k}' to value `{v}`")
            copy_to(config_mask, k, v)
            assert get_val(config_mask, k) == v

    tasks = [{"config": config_mask, "wildcard": []}]
    agents = [{"config": a, "wildcard": [], "filename": k, "path": agent_sources} for k, a in agent_masks.items()]
    log.info(f"Applying `config` settings:")

    for agent_setting in scan_cmd.get("agent", ()):
        # setting:
        #   list of dicts of the form
        #     { key: ...,
        #       val: [...],               # list of values to set `key` in `config_mask`
        #       alias: [...],             # corresponding aliases of `val` for wildcard
        #       wildcard: [key, wildcard],  # wildcard
        #       ref: {...}                # optional referencing for relational values, see below
        #     }
        agents = replace_key_val(agent_setting, agents, self_ref="agent", cross_ref="config")

    for config_setting in scan_cmd.get("config", ()):
        tasks, agents = replace_key_val(config_setting, tasks, self_ref="config", cross_ref="agent", references=agents)

    # GENERATE CONFIG AND AGENT WILDCARDS
    config_destination = os.path.join(destination, config_dst)
    agent_destination = os.path.join(destination, agent_dst)
    config_wildcard, agent_wildcard = get_scan_wildcard_mask(wildcards=scan_cmd["wildcards"],
                                                             config_prefix=config_destination,
                                                             agent_prefix=agent_destination,
                                                             postfix='.yml')

    wildcards = {}

    # EXPORTING CONFIGS
    log.info(f"Exporting {len(tasks)} config-tasks to destination '{config_destination}'")
    for task in tasks:
        filename = copy.copy(config_wildcard)
        for name, wildcard, value in task["wildcard"]:
            filename = filename.replace(wildcard, str(value))

        log.info(f"- config: '{filename}'")
        filedir = os.path.dirname(filename)
        os.makedirs(filedir, exist_ok=True)
        with open(filename, "w") as stream:
            yaml.safe_dump(task["config"], stream, default_flow_style=False, sort_keys=False)

        for name, key, val in task["wildcard"]:
            task_wildcards = wildcards.get((name, key), [])
            if hasattr(val, "__iter__") and not isinstance(val, str):
                task_wildcards.extend(val)
            else:
                task_wildcards.append(val)
            wildcards[(name, key)] = list(set(task_wildcards))

    # EXPORTING AGENTS, DOUBLE OCCURRENCES POSSIBLE
    agent_filenames = []
    for agent in agents:
        filename_mask = []
        prefix = os.path.join(agent_destination, os.path.basename(agent["filename"]).replace('.yml', '-'))
        for name, wildcard in scan_cmd["wildcards"]["agent"]:
            if name + "_" in prefix:
                continue

            for agent_name, _, value in agent["wildcard"]:
                if name == agent_name:
                    filename_mask.append([name, value])
                    break

        filename = get_wildcard_mask(wildcard_mask=filename_mask, prefix=prefix, postfix='.yml')
        filename = filename.replace("-.yml", ".yml")
        if filename in agent_filenames:
            continue

        filedir = os.path.dirname(filename)
        os.makedirs(filedir, exist_ok=True)
        with open(filename, "w") as stream:
            yaml.safe_dump(agent["config"], stream, default_flow_style=False, sort_keys=False)
        agent_filenames.append(filename)

        for name, key, val in agent["wildcard"]:
            task_wildcards = wildcards.get((name, key), [])
            if hasattr(val, "__iter__") and not isinstance(val, str):
                task_wildcards.extend(val)
            else:
                task_wildcards.append(val)
            wildcards[(name, key)] = list(set(task_wildcards))

    # PRINT EXPORTED AGENT FILENAMES
    log.info(f"Exporting {len(agent_filenames)} agents to destination '{agent_destination}'")
    for filename in agent_filenames:
        log.info(f"- agent: '{filename}'")

    # PRINT WILDCARD NAMES
    log.info(f"Filename wildcards:")
    log.info(f"- config: '{config_wildcard}'")
    log.info(f"- agent:  '{agent_wildcard}'")

    return config_wildcard, agent_wildcard, wildcards


def get_dst_path(*args, path="data/", agent="agent.yml"):
    args_str = [str(a).replace(".json", "").replace(".yml", "") for a in args]
    agent_name = agent.split('/')[-1].replace('.yml', '')
    return os.path.join(path, *args_str, agent_name)


def wildcard_sbatch(agent, config, sbatch, wildcards, script, method,
                    args=(),
                    kwargs=None,
                    destination="data/sbatch_scan/",
                    slurm_prefix="",
                    slurm_preamble="mpirun -np $SLURM_NTASKS",
                    slurm_submit=False,
                    num_runs=1
                    ):
    """ Generate """

    os.makedirs(destination, exist_ok=True)
    log.basicConfig(
        level=log.INFO,
        format="%(asctime)s > %(message)s",
        handlers=[
            log.FileHandler(os.path.join(destination, "scan.log")),
            log.StreamHandler()
        ]
    )

    if isinstance(args, str):
        log.info("Loading `args` json")
        args = json.loads(args)

    if isinstance(kwargs, str):
        log.info("Loading `kwargs` json")
        kwargs = json.loads(kwargs)

    # allow wildcards in agent_path and arc_id, but hook occurrences in multiple file-masks
    if isinstance(wildcards, str):
        log.info("Loading `wildcards` json")
        wildcards = json.loads(wildcards)

    wildcard_hooks = {}
    if wildcards:
        log.info("Filling `wildcards`")
        for n, k, v in wildcards:
            wildcard_pairs = [(agent, config)]
            for wildcard_combination in wildcard_pairs:
                if all(k in file_mask for file_mask in wildcard_combination):
                    wildcard_hooks[n] = [v] if not isinstance(v, (tuple, list)) else list(v)

        agent = fill_pattern(agent, {k: v for n, k, v in wildcards})
        config = fill_pattern(config, {k: v for n, k, v in wildcards})

    agents = [agent] if isinstance(agent, str) else agent
    configs = [config] if isinstance(config, str) else config

    with open(sbatch, 'r') as f:
        sbatch_pattern = f.readlines()

    def get_wildcard(name, value):
        if name == "":
            return value
        return f"{name}_{value}"

    from itertools import product
    for agent, config in product(agents, configs):
        # only allow combinations with same wildcards, if they are present in several patterns
        hooked = True
        for name, hook in wildcard_hooks.items():
            if isinstance(hook, str) or not hasattr(hook, "__iter__") or len(hook) == 1:
                # unique
                continue

            hook = [get_wildcard(name, w) for w in hook]
            if all(all(w in pattern for w in hook) for pattern in (agent, config)):
                # check for non-unique hooks, e.g., "_AB" and "_ABC" in pattern "abc_ABC"
                count_a = [agent.count(w) for w in hook]
                count_c = [config.count(w) for w in hook]
                if all(ca == 1 for ca in count_a) and all(cc == 1 for cc in count_c):
                    # all hooks appear once in both patterns agent and config
                    continue

            for i, hook_i in enumerate(hook):
                for hook_j in hook[i+1:]:
                    if any(hook_i in os.path.basename(pattern) for pattern in (agent, config)):
                        if any(hook_j in os.path.basename(pattern) for pattern in (agent, config)):
                            hooked = False
                            break

                if not hooked:
                    break

        if not hooked:
            continue

        if not os.path.exists(agent):
            continue

        log.info(f"Preparing sbatch for agent `{agent}`.")

        arc_train_call = f"python {script} {method}"
        train_args = (*args, agent)
        arg_str = " ".join(train_args)

        train_kwargs = [("config", config)] + [(k, v) for k, v in kwargs.items() if not isinstance(v, bool) and k != "hyper_params"]
        hparams = []
        if "hyper_params" not in kwargs:
            for name, wildcard, value in wildcards:
                h = [w for w in [get_wildcard(name, w) for w in value] if w in config and w not in os.path.basename(agent) or w in os.path.dirname(agent)]
                if h:  # check for double occurrences, e.g., [noise_0.0, noise_0.01] -> choose by longer strings
                    hl = [len(hi) for hi in h]
                    hparams.append(h[argmax(hl)])
        if hparams or "hyper_params" in kwargs:
            hparams = "-".join((*kwargs.get("hyper_params", []), *hparams))
            train_kwargs += [("hyper_params", hparams)]
        kw_str = " ".join([f"--{k.replace('_', '-')} {v}" for k, v in train_kwargs])

        flag_args = [(k, v) for k, v in kwargs.items() if isinstance(v, bool)]
        flag_str = " ".join([f"--{k.replace('_', '-')}" for k, v in flag_args if v])

        arc_train_call = " ".join([slurm_preamble, arc_train_call, arg_str, kw_str, flag_str])

        sbatch_content = []
        added_call = False
        for line in sbatch_pattern:
            if not line.startswith(slurm_preamble):
                sbatch_content.append(line)
                continue
            else:
                for _ in range(num_runs):
                    sbatch_content.append(arc_train_call + "\n")
                added_call = True

        if not added_call:
            for _ in range(num_runs):
                sbatch_content.append(arc_train_call + "\n")

        dst_args = args if not hparams else [*args, hparams]  # TODO: hparams!
        dst_path = get_dst_path(*dst_args, path=destination, agent=agent)
        os.makedirs(dst_path, exist_ok=True)

        prefix = slurm_prefix + '-' * (len(slurm_prefix) > 0)
        sbatch_dst = os.path.join(dst_path, prefix + os.path.basename(sbatch)).replace("//", "/")
        with open(sbatch_dst, "w") as f:
            f.writelines(sbatch_content)

        sbatch_exec = f"sbatch {sbatch_dst}"
        log.info(f"- exec with: `{sbatch_exec}`")
        log.info(f"  from wd:   `{os.getcwd()}`")
        if slurm_submit:
            os.system(sbatch_exec)
            log.info(f"  submitted")
    log.info(f"Done")


if __name__ == '__main__':
    destination = "/home/bene/Projects/mindcraft/tests/data/test/mindcraft_test/util/hpc/"
    c, a = dump_scan(scan_config="/home/bene/Projects/mindcraft/tests/mindcraft_test/util/hpc/scan_instructions.yml",
                     config_source="/home/bene/Projects/mindcraft/tests/mindcraft_test/util/hpc/config.yml",
                     agent_sources="/home/bene/Projects/mindcraft/tests/mindcraft_test/util/hpc/agents/",
                     destination=destination,
                     dev=False,
                     )

    wildcard_sbatch(agent=a, config=c,
                    sbatch="/home/bene/Projects/mindcraft/tests/mindcraft_test/util/hpc/mask.sbatch",
                    wildcards=[("embd", "{EMBD}", ["flatten", "mean"]),
                               ("state", "{STATE}", ["A3T3", "A3TeH1R1"]),
                               ("grid", "{GRID}", ["crs", "sqr"]),
                               ("es", "{ES}", ["CMAES", "SimpleGA"]),
                               ],
                    script="/home/bene/Projects/basal-cognition/examples/arc/arc.py",
                    method="train",
                    args=("czech_4x4.json",),
                    kwargs={"dataset": "/home/bene/Projects/basal-cognition/examples/arc/dataset/BACO-tasks", "new_model": True, "path": destination},
                    destination=destination,
                    )
