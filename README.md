# Natural Language Programming: Sentiment Analysis
## Overview
This project was to develop a few different classifiers to determine the sentiment of a text (positive, negative, or neutral).

The different files in this project are:
- The [Data](Data/) folder contains all of the training and testing data.
- The [classify.py](classify.py) file contains the functions to create, train, and use the different classifiers.
- The [score.py](score.py) file contains the functions to analyze the results from the different classifiers against the true data.
- The [main.ipynb](main.ipynb) file shows the code being run in a Jupyter Notebook with the results from the different classifiers.

## Data
The data was separated out into a few different files:
- [Data/pos-words.txt](Data/pos-words.txt): Contains tokens that are considered positive for training.
- [Data/neg-words.txt](Data/neg-words.txt): Contains tokens that are considered negative for training.
- [Data/train.docs.txt](Data/train.docs.txt): Contains the training documents, one sample per line.
- [Data/train.classes.txt](Data/train.classes.txt): Contains the classes that each corresponding training document is, one sample per line.
- [Data/test.docs.txt](Data/test.docs.txt): Contains the testing documents, one sample per line.
- [Data/test.classes.txt](Data/test.classes.txt): Contains the classes that each corresponding testing document is, one sample per line.

### Pre-Processing

The data files are read in and the samples are split properly so they can be fed in to the models easily. The text for the documents also has a simple tokenizer applied that case-folds the text (makes all text lowercase), removes trailing or leading non-alphanumeric characters for each token, and makes sure each token has at least 1 alphanumeric character.

## Models
 The different types of classifiers are:
- Baseline: Just predict the class as the most common one in the training set.
- Lexicon: Given a pre-defined list of positive and negative words, the text is predicted based on which type of word is used more.
- Logistic Regression: Makes a logistic regression classifier from sklearn.
- Naive Bayes: Uses the Naive Bayes multinomial algorithm with Laplace add 1 smoothing and transforms to logarithmic scale.
- Binarized Naive Bayes: A modified version of the above classifier where each token per document is only used once.

See [classify.py](classify.py) for the implementations of the different classifiers, how they train, and how they predict the class for the text.

## Results

For a general overview of the different models, here is the general accuracy scores for each model on the testing documents:

|         Model         | Testing Accuracy |
|:---------------------:|:----------------:|
|        Baseline       |       0.67       |
|        Lexicon        |       0.44       |
|  Logistic Regression  |        0.7       |
|      Naive Bayes      |       0.708      |
| Binarized Naive Bayes |       0.713      |

See [main.ipynb](main.ipynb#scores) for a more thourough breakdown of the statistics for each model.
