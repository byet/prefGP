# prefGP

`prefGP` is a Gaussian process based library for learning from preference and choice data. It is implemented using Jax and PyTorch. 
`prefGP` implements 9 models to learn from preference and choice data:

* Model 1: Consistent Preferences.
* Model 2: Just Noticeable Difference 
* Model 3: Probit for Erroneous Preferences
* Model 4: Preferences with Gaussian noise error
* Model 5: Probit  for Erroneous preferences as a classification problem
* Model 6: Thurstonian model for label preferences
* Model 7: Plackett-Luce model for label ordering data
* Model 8: Paired comparison for label preferences
* Model 9: Rational and Pseudo-rational models for choice data

## Installation

**Requirements**:

* Python >= 3.11

Download the repository  and then install

```bash
pip install -r requirements.txt
```
## Example
The `notebooks` folder includes several ipython notebooks that demonstrate the use of prefGP. For more details about the models used in the examples, please see below paper.

## Citing Us


## The Team
The library was developed by
- [Alessio Benavoli](https://alessiobenavoli.com/) (Trinity College Dublin)
- [Dario Azzimonti](https://sites.google.com/view/darioazzimonti/home) (IDSIA)
