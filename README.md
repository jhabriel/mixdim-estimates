# mdestimates: A posteriori error estimates for mixed dimensional elliptic equations

**mdestimates** computes a posteriori error estimates for the incompressible flow in fractured porous media. This python package relies heavily on [PorePy](https://github.com/pmgbergen/porepy) and [quadpy](https://github.com/nschloe/quadpy).

## Citing
Add arXiv

## Installation from source

**mdestimates** is developed under Python >= 3.6. Get the latest version using git, i.e.:

    git clone https://github.com/jhabriel/mixdim-estimates.git
    cd mixdim-estimates
  
Install the dependencies:
 
     pip install -r requirements.txt
     
Note that we assume that PorePy >= 1.2.6 is installed. If you do not have PorePy installed, an error message will be printed out.
 
To install the package:

    pip install .

Or, for user-editable installations, 

    pip install --editable .

## Getting started

A simple usage of **mdestimates** can be found in tutorials/basic_tutorial.ipynb.

## Examples

All the numerical examples included in REF can be found under the paper_examples folder.

## Problems?
Create an [issue](https://github.com/jhabriel/mixdim-estimates)

## License
See [license md](./LICENSE.md).
