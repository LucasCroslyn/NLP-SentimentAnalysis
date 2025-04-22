# Natural Language Programming: Sentiment Analysis

This project was to develop a few different classifiers to determine the sentiment of a text (positive, negative, or neutral). The different types of classifiers are:
- Baseline: Just predict the class as the most common one in the training set.
- Lexicon: Given a pre-defined list of positive and negative words, the text is predicted based on which type of word is used more.
- Logistic Regression: Makes a logistic regression classifier from sklearn.
- Naive Bayes: Uses the Naive Bayes multinomial algorithm with Laplace add 1 smoothing and transforms to logarithmic scale.
- Binarized Naive Bayes: A modified version of the above classifier where each token per document is only used once.

- The [Data](Data/) folder contains all of the training and testing data.
- The [classify.py]() file contains the functions to create, train, and use the different classifiers.
- The [score.py]() file contains the functions to analyze the results from the different classifiers against the true data.
- The [main.ipynb]() file shows the code being run in a Jupyter Notebook with the results from the different classifiers.
