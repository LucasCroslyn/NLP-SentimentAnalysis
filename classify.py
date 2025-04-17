#
# Based on code from Dr. Paul Cook, UNB
#

import math, re
import sys
# Do not use the following libraries for your code
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# A simple tokenizer. Applies case folding
def tokenize(s):
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search('\w', t):
            # t contains at least 1 alphanumeric character
            t = re.sub('^\W*', '', t)  # trim leading non-alphanumeric chars
            t = re.sub('\W*$', '', t)  # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens


# A most-frequent class baseline
class Baseline:
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.mfc = sorted(klass_freqs, reverse=True,
                          key=lambda x: klass_freqs[x])[0]

    def classify(self, test_instance):
        return self.mfc


# A logistic regression baseline
class LogReg:
    def __init__(self, texts, klasses):
        self.train(texts, klasses)

    def train(self, train_texts, train_klasses):
        # sklearn provides functionality for tokenizing text and
        # extracting features from it. This uses the tokenize function
        # defined above for tokenization (as opposed to sklearn's
        # default tokenization) so the results can be more easily
        # compared with those using NB.
        # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        self.count_vectorizer = CountVectorizer(analyzer=tokenize)
        # train_counts will be a DxV matrix where D is the number of
        # training documents and V is the number of types in the
        # training documents. Each cell in the matrix indicates the
        # frequency (count) of a type in a document.
        self.train_counts = self.count_vectorizer.fit_transform(train_texts)
        # Train a logistic regression classifier on the training
        # data. A wide range of options are available. This does
        # something similar to what we saw in class, i.e., multinomial
        # logistic regression (multi_class='multinomial') using
        # stochastic average gradient descent (solver='sag') with L2
        # regularization (penalty='l2'). The maximum number of
        # iterations is set to 1000 (max_iter=1000) to allow the model
        # to converge. The random_state is set to 0 (an arbitrarily
        # chosen number) to help ensure results are consistent from
        # run to run.
        # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        self.lr = LogisticRegression(multi_class='multinomial',
                                     solver='sag',
                                     penalty='l2',
                                     max_iter=1000,
                                     random_state=0)
        self.clf = self.lr.fit(self.train_counts, train_klasses)

    def classify(self, test_instance):
        # Transform the test documents into a DxV matrix, similar to
        # that for the training documents, where D is the number of
        # test documents, and V is the number of types in the training
        # documents.
        # test_counts = self.count_vectorizer.transform(test_texts)
        test_count = self.count_vectorizer.transform([test_instance])
        # Predict the class for each test document  
        # results = self.clf.predict(test_counts)
        return self.clf.predict(test_count)[0]


# Implement the lexicon-based baseline
# You may change the parameters to each function
class lexicon:
    def __init__(self, pos_words, neg_words):
        self.posneg_words = {}
        self.train(pos_words, neg_words)

    def train(self, pos_words, neg_words):
        # Make a dictionary pos/neg words, key is the word and value is 1 if positive and -1 if negative
        self.posneg_words.update({x.strip(): 1 for x in open(pos_words, encoding='utf8')})
        self.posneg_words.update({x.strip(): -1 for x in open(neg_words, encoding='utf8')})

    def classify(self, test_instance):
        value = 0
        tokenized_instance = tokenize(test_instance)

        # For each token in the instance,
        # if pos word +1 to value, if neg word -1 to word, if neutral word just +0 to value
        for token in tokenized_instance:
            value += self.posneg_words.get(token, 0)
        if value > 0:
            return "positive"
        elif value < 0:
            return "negative"
        elif value == 0:
            return "neutral"


# Implement the multinomial Naive Bayes model with smoothing
class NaiveBayes:
    def __init__(self, texts, classes):
        self.word_pos_freq = {}
        self.word_neg_freq = {}
        self.word_neu_freq = {}

        self.word_pos_prob = {}
        self.word_neg_prob = {}
        self.word_neu_prob = {}

        self.unique_tokens: set = set()

        self.class_prob = {}
        self.train(texts, classes)

    def train(self, train_texts, train_klasses):
        # Calculates the # of docs in each class
        klass_freqs = {}
        for k in train_klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        # Calculates the prob of a doc being a class
        # log(# docs in class k / # docs total)
        for k in klass_freqs:
            self.class_prob[k] = math.log(klass_freqs[k] / len(train_klasses))

        for index in range(len(train_texts)):
            tokenized_doc = tokenize(train_texts[index])
            doc_type = train_klasses[index]
            for token in tokenized_doc:
                if token not in self.unique_tokens:
                    self.unique_tokens.add(token)
                if doc_type == "positive":
                    self.word_pos_freq[token] = self.word_pos_freq.get(token, 0) + 1
                elif doc_type == "negative":
                    self.word_neg_freq[token] = self.word_neg_freq.get(token, 0) + 1
                elif doc_type == "neutral":
                    self.word_neu_freq[token] = self.word_neu_freq.get(token, 0) + 1

        # Go through all the unique tokens and calculate their prob of appearing for each class
        # Freq token appeared in training docs of class c + 1 / total tokens in docs of class c + # unique tokens
        for token in self.unique_tokens:
            self.word_pos_prob[token] = math.log((self.word_pos_freq.get(token, 0) + 1) /
                                                 (sum(self.word_pos_freq.values()) + len(self.unique_tokens)))
            self.word_neg_prob[token] = math.log((self.word_neg_freq.get(token, 0) + 1) /
                                                 (sum(self.word_neg_freq.values()) + len(self.unique_tokens)))
            self.word_neu_prob[token] = math.log((self.word_neu_freq.get(token, 0) + 1) /
                                                 (sum(self.word_neu_freq.values()) + len(self.unique_tokens)))

    def classify(self, test_instance):
        class_options = {"positive": self.class_prob.get("positive", -math.inf),
                         "negative": self.class_prob.get("negative", -math.inf),
                         "neutral": self.class_prob.get("neutral", -math.inf)}

        for token in tokenize(test_instance):
            # If the test token has been seen in the training data,
            # get the previously calculated prob of that token for the class

            # Else, calculate prob of token with freq of 1 (since smoothing) /
            # tokens in docs of class c + # unique tokens

            if token in self.unique_tokens:
                ttoken_prob_pos = self.word_pos_prob[token]
                ttoken_prob_neg = self.word_neg_prob[token]
                ttoken_prob_neu = self.word_neu_prob[token]

            else:
                ttoken_prob_pos = math.log(1 / (sum(self.word_pos_freq.values()) + len(self.unique_tokens)))
                ttoken_prob_neg = math.log(1 / (sum(self.word_neg_freq.values()) + len(self.unique_tokens)))
                ttoken_prob_neu = math.log(1 / (sum(self.word_neu_freq.values()) + len(self.unique_tokens)))

            # Sum of the test doc's token probs + prior prob of class for all possible classes
            class_options.update({"positive": class_options["positive"] + ttoken_prob_pos})
            class_options.update({"negative": class_options["negative"] + ttoken_prob_neg})
            class_options.update({"neutral": class_options["neutral"] + ttoken_prob_neu})

        # Getting key with the max value code adapted from here https://stackoverflow.com/a/280156
        return max(class_options, key=class_options.get)


##Implement the binarized multinomial Naive Bayes model with smoothing
class BinaryNaiveBayes:
    def __init__(self, texts, classes):
        self.word_pos_freq = {}
        self.word_neg_freq = {}
        self.word_neu_freq = {}

        self.word_pos_prob = {}
        self.word_neg_prob = {}
        self.word_neu_prob = {}

        self.unique_tokens: set = set()

        self.class_prob = {}
        self.train(texts, classes)

    def train(self, train_texts, train_klasses):
        # Calculates the # of docs in each class
        klass_freqs = {}
        for k in train_klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        # Calculates the prob of a doc being a class
        # log(# docs in class k / # docs total)
        for k in klass_freqs:
            self.class_prob[k] = math.log(klass_freqs[k] / len(train_klasses))

        for index in range(len(train_texts)):
            tokenized_doc = tokenize(train_texts[index])

            doc_set = set()
            doc_type = train_klasses[index]

            for token in tokenized_doc:
                if token not in self.unique_tokens:
                    self.unique_tokens.add(token)

                # Only add token to count if not seen in the document's dictionary previously
                if doc_type == "positive":
                    if token not in doc_set:
                        doc_set.add(token)
                        self.word_pos_freq[token] = self.word_pos_freq.get(token, 0) + 1

                elif doc_type == "negative":
                    if token not in doc_set:
                        doc_set.add(token)
                        self.word_neg_freq[token] = self.word_neg_freq.get(token, 0) + 1

                elif doc_type == "neutral":
                    if token not in doc_set:
                        doc_set.add(token)
                        self.word_neu_freq[token] = self.word_neu_freq.get(token, 0) + 1

        # Go through all the unique tokens and calculate their prob of appearing for each class
        # Freq token appeared in training docs of class c + 1 / total tokens in docs of class c + # unique tokens
        for unique_token in self.unique_tokens:
            self.word_pos_prob[unique_token] = math.log((self.word_pos_freq.get(unique_token, 0) + 1) /
                                                        (sum(self.word_pos_freq.values()) + len(self.unique_tokens)))
            self.word_neg_prob[unique_token] = math.log((self.word_neg_freq.get(unique_token, 0) + 1) /
                                                        (sum(self.word_neg_freq.values()) + len(self.unique_tokens)))
            self.word_neu_prob[unique_token] = math.log((self.word_neu_freq.get(unique_token, 0) + 1) /
                                                        (sum(self.word_neu_freq.values()) + len(self.unique_tokens)))

    def classify(self, test_instance):
        class_options = {"positive": self.class_prob.get("positive", -math.inf),
                         "negative": self.class_prob.get("negative", -math.inf),
                         "neutral": self.class_prob.get("neutral", -math.inf)}

        test_instance_unique = set()

        for token in tokenize(test_instance):
            # Each unique token is only used once in the calculation
            if token not in test_instance_unique:
                test_instance_unique.add(token)

                if token in self.unique_tokens:
                    ttoken_prob_pos = self.word_pos_prob[token]
                    ttoken_prob_neg = self.word_neg_prob[token]
                    ttoken_prob_neu = self.word_neu_prob[token]

                else:
                    ttoken_prob_pos = math.log(1 / (sum(self.word_pos_freq.values()) + len(self.unique_tokens)))
                    ttoken_prob_neg = math.log(1 / (sum(self.word_neg_freq.values()) + len(self.unique_tokens)))
                    ttoken_prob_neu = math.log(1 / (sum(self.word_neu_freq.values()) + len(self.unique_tokens)))

                # Sum of the test doc's token probs + prior prob of class for all possible classes
                class_options.update({"positive": class_options["positive"] + ttoken_prob_pos})
                class_options.update({"negative": class_options["negative"] + ttoken_prob_neg})
                class_options.update({"neutral": class_options["neutral"] + ttoken_prob_neu})

        # Getting key with the max value code from here https://stackoverflow.com/a/280156
        return max(class_options, key=class_options.get)


if __name__ == '__main__':

    sys.stdout.reconfigure(encoding='utf-8')

    # Method will be one of 'baseline', 'lr', 'lexicon', 'nb', or 'nbbin'

    method = sys.argv[1]

    train_texts_fname = sys.argv[2]
    train_klasses_fname = sys.argv[3]
    test_texts_fname = sys.argv[4]

    train_texts = [x.strip() for x in open(train_texts_fname,
                                           encoding='utf8')]
    train_klasses = [x.strip() for x in open(train_klasses_fname,
                                             encoding='utf8')]
    test_texts = [x.strip() for x in open(test_texts_fname,
                                          encoding='utf8')]

    # Check which method is being asked to implement from user
    if method == 'baseline':
        classifier = Baseline(train_klasses)

    elif method == 'lr':
        # Use sklearn's implementation of logistic regression
        classifier = LogReg(train_texts, train_klasses)

    elif method == 'lexicon':
        # Should get accuracy of 0.44
        classifier = lexicon("pos-words.txt", "neg-words.txt")

    elif method == 'nb':
        # Should get accuracy of 0.708
        classifier = NaiveBayes(train_texts, train_klasses)

    elif method == 'nbbin':
        # Should get accuracy of .715/.713
        classifier = BinaryNaiveBayes(train_texts, train_klasses)

    else:
        classifier = None
    assert classifier is not None

    # Run the classify method for each instance
    results = [classifier.classify(x) for x in test_texts]

    # Create output file at given output file name
    # Store predictions in output file
    outFile = sys.argv[5]
    out = open(outFile, 'w', encoding='utf-8')
    for r in results:
        out.write(r + '\n')
    out.close()