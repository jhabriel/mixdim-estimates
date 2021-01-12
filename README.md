# mdestimates: A posteriori error estimates for mixed dimensional elliptic equations

**mdestimates** is a Python package created for computing *a posteriori* error estimates for mixed-dimensional elliptic equations. That is, the set of equations that models the flow in fractured porous media. The package is build as an extension of [PorePy](https://github.com/pmgbergen/porepy). Note that **mdestimates** also relies on the numerical integration package [quadpy](https://github.com/nschloe/quadpy).

## Citing

If you use **mdestimates** in your research, we ask you to cite the following reference:
*Add arXiv*

## Installation from source

**mdestimates** is developed under Python >= 3.6. Get the latest version using by cloning this git repo, i.e.:

    git clone https://github.com/jhabriel/mixdim-estimates.git
    cd mixdim-estimates
  
Now, install the dependencies:
 
     pip install -r requirements.txt
     
Note that we assume that PorePy >= 1.2.6 is installed. If you do not have PorePy installed, please do so before installing the dependencies.
 
To install the **mdestimates**:

    pip install .

Or, for user-editable installations:

    pip install --editable .

## Getting started

A simple usage of **mdestimates** can be found in tutorials/basic_tutorial.ipynb.

## Examples

All the numerical examples included in REF can be found under the paper_examples folder. These includes two validation cases and two benchmark problems.

## Problems, suggestions, enhancements...
Create an [issue](https://github.com/jhabriel/mixdim-estimates).

## License
See [license md](./LICENSE.md).
