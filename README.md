# Partially-Interpretable-Nearest-Neighbor
A code for the thesis "Partially Interpretable Nearest Neighbor: Combining Nearest Neighbor with a Black-box Model".

## How to use
`hybrids.py` contains all the core functions and classes, including the implementation of hybrid nearest neighbors.

All other Jupyter notebooks and an R file run experiments that were also conducted for the thesis (except for Bonus), serving as examples of these functions and classes.

These examples and the docstrings for each function and class should be enough to understand how to use them.

### Datasets and black boxes
Only the Census Income datasets from the UCI machine learning repository are provided to run these examples, along with a prediction
of them by a random forest black-box model. If you want to run the examples with other datasets or black-box models (the commented-out lines in the first few cells in all notebooks),
you need to download these datasets using URLs in `process_data.ipynb` and fit the corresponding black-box model using `blackboxes.ipynb`.