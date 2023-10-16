# **F<small>IORA</small>**

> **Disclaimer: Early Development / Prototype Notice**<br>
> **F<small>IORA</small>** is an algorithm in its early stages of development and is provided as a prototype.
Performance is not guaranteed, functionality may be incomplete, and usability was not a central concern during this phase of development. 
Users should exercise caution.

**F<small>IORA</small>** is an *in silico* fragmentation algorithm for small compounds and produces simulated tandem mass spectra (MS/MS). The framework uses a graph neural network as the core module and edge prediction to identify likely bond cleavages and fragment ion intensities. Additionally, **F<small>IORA</small>** predicts retention time (RT) and collision cross section (CCS) of the compounds.

## Requirements

Developed and tested tested with the following system and version:

* Debian GNU/Linux 11 (bullseye)
* Python 3.10.8
* GCC 11.2.0


## Installation

(Optional: Create a new conda environment)

    conda create -n fiora python=3.10.8
    conda activate fiora

Cd into this directory. Then, install package by using the setup.py 

    pip install .


## Usage

### MS/MS prediction

Use spectral prediction function as follows:

    fiora-predict [-h] -i INPUT -o OUTPUT [--model MODEL] [--rt | --no-rt] [--ccs | --no-ccs] [--annotation | --no-annotation]

An input csv file must be provided and an output file specified (mgf or msp format).

## Input format

Input files are expected to be in csv format. With a header defining the columns: "Name", "SMILES", "Precursor_type", "CE", "Instrument_type" and rows listing individual queries.
See example [input file](examples/example_input.csv).

## Output format

Currently, only the mgf format is supported.