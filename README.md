# **F<span style="font-variant:small-caps;">iora</span>**

> **Disclaimer: Early Development / Prototype Notice**<br>
> **F<span style="font-variant:small-caps;">iora</span>** is an algorithm in its early stages of development and is provided as a prototype.
Performance is not guaranteed, functionality may be incomplete, and usability was not a central concern during this phase of development. 
Users should exercise caution.

**F<span style="font-variant:small-caps;">iora</span>** is an *in silico* fragmentation algorithm for small compounds and produces simulated tandem mass spectra (MS/MS). The framework uses a graph neural network as the core module and edge prediction to identify likely bond cleavages and fragment ion intensities. Additionally, **F<span style="font-variant:small-caps;">iora</span>** predicts retention time (RT) and collision cross section (CCS) of the compounds.

## Requirements

Developed and tested with the following systems and versions:

* Debian GNU/Linux 11 (bullseye)
* Python 3.10.8
* GCC 11.2.0


## Installation

Installation guide of the Fiora python package (under 10 minutes):

Clone the project folder 

    git clone https://github.com/BAMeScience/fiora.git

(Optional: Create a new conda environment)

    conda create -n fiora python=3.10.8
    conda activate fiora

Change into the project directory (*cd fiora*). Then, install the package by using the setup.py via

    pip install .

(Optional: You may want to test that the package works as intended. This can be done by running the sripts in the *tests* directory or with pytest (requires: *pip install pytest*))

    pytest -v tests

## Usage

### MS/MS prediction

Use spectral prediction function as follows:

    fiora-predict [-h] -i INPUT -o OUTPUT [--model MODEL] [--rt | --no-rt] [--ccs | --no-ccs] [--annotation | --no-annotation]

An input csv file must be provided and an output file specified (mgf or msp format).

#### Input format

Input files are expected to be in csv format. With a header defining the columns: "Name", "SMILES", "Precursor_type", "CE", "Instrument_type" and rows listing individual queries.
See example [input file](examples/example_input.csv).

#### Output format

Predicted spectra are provided in standard *msp* and *mgf* formats.

#### Example usage

Run the fiora-predict from within the directory

    fiora-predict -i examples/example_input.csv  -o examples/example_spec.mgf

Note that a default model is currently not implemented. We will provide default open-source model weights in the near future. The predictions should only take a few seconds. Specify a GPU device by using the *--dev* option (e.g., --dev cuda:0) for significant speed up.  