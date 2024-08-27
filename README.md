# **F<span style="font-variant:small-caps;">iora</span>**

> **Disclaimer: Early Development / Prototype Notice**<br>
> **F<span style="font-variant:small-caps;">iora</span>** is an algorithm in its early stages of development and is provided as a prototype.
Performance is not guaranteed, functionality may be incomplete, and usability was not a central concern during this phase of development. 
Users should exercise caution.

**F<span style="font-variant:small-caps;">iora</span>** is an *in silico* fragmentation algorithm for small compounds that produces simulated tandem mass spectra (MS/MS). The framework employs a graph neural network to predict bond cleavages and fragment ion intensities via edge prediction. Additionally, **F<span style="font-variant:small-caps;">iora</span>** can estimate retention times (RT) and collision cross sections (CCS) of the compounds.

## Requirements

Developed and tested with the following systems and versions:
* Debian GNU/Linux 11 (bullseye)
* Python 3.10.8
* GCC 11.2.0


## Installation

Installation guide for the Fiora Python package (under 10 minutes):

Clone the project folder 

    git clone https://github.com/BAMeScience/fiora.git

(Optional) Create a new conda environment

    conda create -n fiora python=3.10.8
    conda activate fiora

Change into the project directory (`cd fiora`). Then, install the package by using the setup.py via

    pip install .

(Optional) You may want to test that the package works as intended. This can be done by running the sripts in the *tests* directory or by using pytest (requires: `pip install pytest`)

    pytest -v tests

## Usage

### MS/MS prediction

Use spectral prediction function as follows:

    fiora-predict [-h] -i INPUT -o OUTPUT [--model MODEL] [--rt | --no-rt] [--ccs | --no-ccs] [--annotation | --no-annotation]

An input csv file must be provided and an output file specified (`mgf` or `msp` format).

### Input format

Input files are expected to be in csv format. With a header defining the columns: "Name", "SMILES", "Precursor_type", "CE", "Instrument_type" and rows listing individual queries.
See example [input file](examples/example_input.csv).

### Output format

Predicted spectra are provided in standard `msp` and `mgf` format.

### Example usage

Run the fiora-predict from within this directory

    fiora-predict -i examples/example_input.csv  -o examples/example_spec.mgf

By default, an open-source model is selected automatically, and predictions typically complete within a few seconds. For faster performance, specify a GPU device using the `--dev` option (e.g., `--dev cuda:0`). The output file (e.g., examples/example_spec.mgf) can be compared with the [expected results](examples/expected_output.mgf) to verify model accuracy. This verification is automatically performed by running pytest (as described above).