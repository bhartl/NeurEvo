# Mindcraft 
A framework for cognitive evolution.

Predominantly used for evolutionary reinforcement learning, especially with unconventional neural network architectures.

> **_NOTE:_**  The `mindcraft` framework is under active development and the API might change in the future. The framework is not fully documented, but we are working on it.

> **_WARNING:_**  The `m̀incraft.io.Repr` class is used to represent native classes as yml files, which might also use `eval` to evaluate the class.


## Project Structure
- The entire `mindcraft` framework implementation is located in the [mindcraft](mindcraft) folder.
- General code examples are located in the [examples](examples) directory.
- Examples, such as solving the [_cart-pole_](examples/agents/gym/classic_control/) environment with `mindcraft` are 
  located in the [examples/agents](examples/agents) directory.
- (Unit)tests are implemented in the [tests](tests) folder.

## Install
We install `mindcraft` in a dedicated *virtual environment* usind [Anaconda](https://anaconda.com).
All commands are executed in the project root directory `$PROJECT_ROOT`, which we set as follows
```bash
export PROJECT_ROOT=~/Projects/NeurEvo
```
Please exchange `~/Projects` with the path to your project directory.
If you are working within the `mindcraft` project root directory, you can also use `.` as the project root directory.

### Virtual Environment

The `mindcraft` package should be executed in dedicated *virtual environments (venv)* to not compromise the system's *Python* installation.
With *Anaconda*, a corresponding *mindcraft venv* can be generated as follows
```bash
conda create -n mindcraft python=3.8
```
which can be activated via
```bash
conda activate mindcraft
```
and deactivated by
```bash
conda deactivate
```
 
### Install 
To install the `mindcraft` framework, we use the [pip](https://pip.pypa.io/en/stable/) package manager.
The `mindcraft` framework is installed in *editable* mode, such that changes to the source code are immediately available (c.f., the `-e` option e.g. [here](https://pip.pypa.io/en/stable/cli/pip_install/)).

#### GPU
If a GPU is available, the `mindcraft` framework can be installed via
```bash
export PROJECT_ROOT=~/Projects/NeurEvo
pip install -e $PROJECT_ROOT
```

#### CPU
If no GPU is available, the `mindcraft` framework can be installed via
```bash
pip install -e $PROJECT_ROOT --extra-index-url https://download.pytorch.org/whl/cpu
```

#### Dependencies LOCAL
To enable the MPI multiprocessing framework using [Anaconda-Python](https://anaconda.com), and to install `swig` (necessary for `box2d-py`), the `conda` command can be used

```bash
conda install mpi4py swig
```

Alternatively, the `mpich` package can be installed via

```bash
sudo apt install mpich swig
```

and `mpi4py` can be installed via `pip install -e $PROJECT_ROOT[acc]`.

#### Issues
Also, `evdev` might be necessary:
```bash
sudo apt-get install python3-evdev python3-dev
```

#### Dependencies CLUSTER: `mpi4py`
From [here](https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py): 
[`mpi4py`](https://mpi4py.readthedocs.io/en/stable/) 
provides a Python interface to MPI or the _Message-Passing Interface_. 
It is useful for parallelizing Python scripts. 
Also be aware of multiprocessing, dask and Slurm job arrays.

On the cluster, the `mpich` package could either be installed in a virtual environment via `conda`

```bash
conda install mpi4py swig
```

but is it's **recommended** to use the **system's `mpich` installation**, which can be loaded via the `module` command.
Please checkout, which `mpich` version is available on your system and load the corresponding module.
The command
```bash
module avail | grep openmpi
```
can be used to list all available modules, e.g., showing `openmpi/4.1.4-gcc-<x.y.z>`, where `<x.y.z>` is a version number.

The corresponding **module** can be **loaded** via
```bash
module load openmpi/4.1.4-gcc-<x.y.z>
```
and, **subsequently**, **`mpi4py`** can be **installed** via
```bash
pip install mpi4py --no-cache-dir
```

To **test** the `mpi4py` installation, a simle test script is provided in [tests/test_mpi4py.py](tests/test_mpi4py.py):
a loop of 4 iterations is executed, each prints the rank and the batch `(i/4)` of the current data, then waits for 2 seconds.
To run is **serially**, use
```bash
python tests/test_mpi4py.py
```
which should roughly take **8 seconds to complete**,
and to run it in **parallel**, use
```bash
mpirun -np 4 python tests/test_mpi4py.py
```
which should roughly take **2 seconds to complete**.

> **_NOTE:_**  The `module load openmpi/4.1.4-gcc-<x.y.z>` command needs to be executed **every time** before running `mpirun`. 

#### Gym Rendering Issues
In case rendering issues occur in the `gym` environments, the following packages suggested by [stack overflow](https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris) might be helpful:
```bash
conda install -c conda-forge libstdcxx-ng
```

### Dependencies for Examples and Jupyter Notebooks
To  run the examples, analysis tools and Jupyter notebooks, the `examples` packages listed in the `project.optional-dependencies` of the [pyproject.toml](pyproject.toml) file are required.
These can be installed via
```bash
pip install -e $PROJECT_ROOT[examples]
```

Also, most of the Jupyter examples were evaluated using the `mindcraft` venv. 
Thus, it is recommended to install the `mindcraft` venv as a Jupyter kernel as follows
```bash
ipython kernel install --name "mindcraft" --user
```

## Test
Make sure, be located in the `mindcraft` project root directory:
```bash
cd $PROJECT_ROOT
```
### Run unittests
```bash
python -m unittest discover -s $PROJECT_ROOT/tests
```

### [`CartPole` example](./examples/examples/agents/gym/classic_control/cart_pole/feed_forward) with `feed_forward` policy
For (multicore) **training**, run
```bash
mpirun -np 4 python examples/agents/gym/gymcraft.py train --task classic_control/cart_pole --conf feed_forward --new-model
```

and check the training **progress** with
```bash
tail -f data/examples/agents/gym/classic_control/cart_pole/models/feed_forward/evolved-agent.log
```

**Run** the following command see what the agent is doing

```bash
python examples/agents/gym/gymcraft.py rollout --task classic_control/cart_pole
```


## Authors
- Ben Hartl 

## Citation
If you use `mindcraft` in your research, please cite it as follows:
```bibtex
@misc{hartl2025mindcraft,
  author = {Hartl, Benedikt},
  title = {Mindcraft: A framework for cognitive evolution},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bhartl/neurevo}},
}
```
