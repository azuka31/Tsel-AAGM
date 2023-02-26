# Tsel-AAGM


# Background
Propensity score matching is a statistical technique used to estimate the effect of a treatment or intervention on an outcome of interest. It is commonly used in observational studies, where the assignment of treatment or exposure to a particular group is not randomized.

The idea behind propensity score matching is to balance the characteristics of the treatment and control groups by matching individuals with similar propensity scores, which are the probabilities of receiving the treatment or intervention based on observed covariates. This helps to control for confounding factors and reduce selection bias, allowing for a more accurate estimation of the treatment effect.

Overall, propensity score matching is a useful tool for researchers to make causal inferences in observational studies, although it is important to consider the limitations and assumptions of this method.

# Installation Guide
Will be uploaded to pypi soon,
For temporary usaga, download the matching.py from **tsel_aagm** directory, put on your project directory and import it like the code
```python
from matching import PropensityScoreMatch
```

# Requirements Library
This python requires related package more importantly python_requires='>=3.1', so that package can be install Make sure the other packages meet the requirements below
- pandas>=1.1.5,
- numpy>=1.18.5,
- scipy>=1.2.0,
- matplotlib>=3.1.0,
- statsmodels>=0.8.0

# Usage Guide
This is a Python class named PropensityScoreMatch. It is designed to perform propensity score matching, a technique used to balance the distribution of confounding variables between treatment and control groups in observational studies. The class has four input arguments:

- df: a pandas DataFrame containing the data to be analyzed.
- features: a list of column names in df that contain the variables used to calculate propensity scores.
- treatment: a string that specifies the name of the column in df that contains the treatment variable.
- outcome: a string that specifies the name of the column in df that contains the outcome variable.

The output of the class is two pandas DataFrames:

- df_matched: a DataFrame containing the data for the matched pairs of treated and control observations.
- df_TE: a DataFrame containing the treatment effect estimates for each variable in features.

In addition to these output DataFrames, the class provides two methods for visualizing the results of the analysis:

- plot_smd(): a method that generates a plot of standardized mean differences (SMDs) between the treatment and control groups for each variable in features.
- plot_individual_treatment(): a method that generates a plot of the individual treatment effects for each observation in df_matched.

For Analysis:
- plot_smd() : plotting the df_smd

# Example Usage
Importing libraries
```python
import pandas as pd
import matplotlib.pyplot as plt
from matching import PropensityScoreMatch
pd.set_option('mode.chained_assignment',None)
```
Initiating model
```python
# Importing Data
takers = pd.read_csv('takers_programname.csv').fillna(0)
nontakers = pd.read_csv('nontakers_programname.csv').fillna(0)
takers['label'] = 1
nontakers['label'] = 0
df_main = pd.concat([takers, nontakers], ignore_index=True)

# Defining Variable
features = ['var1','var2','var3']
treatment = 'label'
outcome = 'out1'

# Propensity Score Model
PS_model = PropensityScoreMatch(df_main, features, treatment, outcome)
PS_model.df_TE.head()
```
Output:
![output2](output/df_te.png)

Evaluating Plot
```python
PS_model.plot_smd()
```
Output:

![output2](output/smd.png)
