# The Data Science Util 
[![Latest release](https://img.shields.io/github/v/release/TheJproject/data-science-util?label=Latest%20release&style=social)](https://github.com/TheJproject/data-science-util/releases/tag/v1.0.0)
[![Stars](https://img.shields.io/github/stars/TheJproject/data-science-util?style=social)](https://github.com/TheJproject/data-science-util/stargazers)
[![Fork](https://img.shields.io/github/forks/TheJproject/data-science-util?style=social)](https://github.com/TheJproject/data-science-util/network/members)
[![Watchers](https://img.shields.io/github/watchers/TheJproject/data-science-util?style=social)](https://github.com/TheJproject/data-science-util/watchers)

![Stochastic Parrot](docs/stable-diffusion-xl.jpeg)
<br/>*“Racoon, flat design, vector art” — [Stable Diffusion XL](https://clipdrop.co/stable-diffusion)*



### To-do list

- [ ] Add NAS script
- [ ] Add PCA in feature engineering

    #### Nice to have

- [ ] github action
- [ ] coverage on github
- [ ] make official releases
- [ ] improve quality of life
- [ ] Comet Haley


### Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)
7. [Contact](#contact)

## Introduction

The "Data Science Utils" project aims to provide a collection of data science utilities for managing and processing datasets. Based on the cookiecutter data science template, it follows a standardized structure for organizing data science projects that separates code, data, and results, making the work easy to understand and reproduce. The main idea behind this structure is to separate the code, data, and results, and make it easy for others to understand and reproduce the work.

The project is structured following the Makefile pattern, allowing for automated execution of different stages of the project. The Makefile contains a set of rules defining dependencies and actions for each step. The user can simply execute the make command with the desired target to trigger the corresponding rule.

The general workflow of the program is as follows:

1. **Data preprocessing:** The raw data is preprocessed and cleaned, preparing it for feature engineering and modeling (`make data`).
2. **Feature engineering:** The cleaned data is transformed into a set of features for modeling (`make feature`).
3. **Model training:** The engineered features are used to train a model to predict the target variable (`make train`).
4. **Model ensembling:** Multiple models are combined to improve overall performance (`make ensemble`).
5. **Prediction:** The trained models are used to make predictions on new data (`make predict`).
6. **Kaggle submission:** The predicted values are submitted to the Kaggle competition (`make kaggle_submit`).

The program is designed to be flexible and customizable. Users can modify the configuration file to adjust the parameters of each step, add custom features or external data, and use different models and ensembling techniques. It can be installed as a package using the pip command, and users can start a new project by running the cookiecutter command with the desired options.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Python3 and the following Python libraries are required for this software. Make sure to install them.

```bash
pip install -r requirements.txt
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
