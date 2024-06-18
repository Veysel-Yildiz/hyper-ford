# PyBorg
A pure Python implementation of the Borg MOEA using the [Platypus optimization library](http://github.com/Project-Platypus/Platypus).

This version is best suited for running small experiments on a personal computer or workstation. Due to being written in Python, this code
is significantly less efficient than the C implementation.  This is intended for learning about the Borg MOEA, prototyping changes, and
small-scale optimization.

## Prerequisites
- [Python](https://www.python.org) - Version 3.5 or newer
- [Platypus](http://github.com/Project-Platypus/Platypus) - Install using `pip install platypus-opt`
- [matplotlib](https://matplotlib.org) - Optional for generating figures, install with `pip install matplotlib`

## Usage
Three example files are provided, which perform simple optimizations on the popular DTLZ2 test problem. 

```bash
# Install PyBorg and its dependencies
pip install -U build setuptools platypus-opt matplotlib
python -m build

python dtlz2.py                 # Solve the DTLZ2 test problem and print the results
python dtlz2_plot.py            # Plot the final approximation set
python dtlz2_advanced.py        # Demonstrates setting custom parameters and operators
python dtlz2_runtime.py         # Collects runtime dynamic and generates a plot of the Hypervolume throughout a run
```

## Defining a Problem
Refer to the [Platypus documentation](https://platypus.readthedocs.io/en/latest/) for information on defining problems in Platypus.
Once a problem is defined, create a PyBorg instance for the problem with respective `problem` and `epsilons` values.
```python
borg = BorgMOEA(problem, epsilons)
```

## License
Copyright 2012-2014 The Pennsylvania State University, Copyright 2020 Cornell University

This software was written by Andrew Dircks, Dave Hadka, and others.

The use, modification and distribution of this software is governed by the The Pennsylvania State University Research and Educational Use License.
You should have received a copy of this license along with this program. If not, contact <info@borgmoea.org>.
