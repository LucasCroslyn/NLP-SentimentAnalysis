import math, re

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def tokenize(s):
    '''
    A simple word tokenizer. Applies case folding. Removes leading and trailing non-alphanumeric characters including punctuation for each word (token) in the text. Keeps any other non-alphanumeric characters within the word (token).

    :param s: The input string to tokenize into separate words (tokens).
    :return: Returns a list of all the words (tokens) from the input string. 
    '''
    
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search(r'\w', t): # t contains at least 1 alphanumeric character
            t = re.sub(r'^\W*', '', t)  # trim leading non-alphanumeric chars
            t = re.sub(r'\W*$', '', t)  # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens


class Baseline:
    '''
    A baseline sentiment classifier. Will always classify text as the most common class.
    '''
    
    def __init__(self, train_classes):
        '''
        Initializes the sentiment classifier. Must be given the training data.

        :param train_classes: A list of what each training sample's class is (e.g. 'negative', 'positive', 'negative', 'neutral', 'positive', etc.)
        '''
        self.train(train_classes)

    
    def train(self, train_classes):
        '''
        Determines the counts for each class based on the training data. Stores the most common class.

        :param train_classes: A list of what each training sample's class is (e.g. 'negative', 'positive', 'negative', 'neutral', 'positive', etc.)
        '''
        
        class_freqs = Counter(train_classes)
        self.freq_class = class_freqs.most_common(1)[0][0]

    
    def classify(self, test_instance):
        '''
        Classifies some input text. Will always classify it as the most common class based on the training data.

        :param test_instance: The text to classify. Doesn't actually use it for anything.
        :return: Returns the classification for the text. Will be the most common class from the training data.
        '''
        return self.freq_class


class LogReg:
    '''
    A logistic regression based sentiment classifier.
    '''
    def __init__(self, texts, classes):
        '''
        Initializes the logistic regression sentiment classifier. Must be given the training text and classes.

        :param texts: A list of all of the text being trained on.
        :param classes: A list of each class for each training sample.
        '''
        self.train(texts, classes)

    
    def train(self, train_texts, train_classes):
        '''
        Trains the logistic regression sentiment classifier with training text and classes.

        :param train_texts: A list of all of the texts being trained on.
        :param train_classes: A list of each class for each training sample.
        '''
        
        # Won't be using the default tokenizer for sklearn, specific one made above (a 'weaker', more barebones one)
        self.count_vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None)
        
        # train_counts will be a DxV matrix where D is the number of training documents. 
        # V is the number of types in the training documents. 
        # Each cell in the matrix indicates the frequency (count) of a type in a document.
        self.train_counts = self.count_vectorizer.fit_transform(train_texts)
        
        # Train a logistic regression classifier on the training data. 
        # Multinomial logistic regression using
        # stochastic average gradient descent (solver='sag') with 
        # L2 regularization (penalty='l2'). 
        # The maximum number of iterations is set to 1000 (max_iter=1000) to allow the model to converge. 
        # The random_state is set to 0 (an arbitrarily chosen number) to help ensure results are consistent from run to run.
        lr = LogisticRegression(solver='sag',
                                     penalty='l2',
                                     max_iter=1000,
                                     random_state=0)
        
        # Trains the logistic regression on the input training data
        self.trained_lf = lr.fit(self.train_counts, train_classes)

    
    def classify(self, test_instance):
        '''
        Classifies some input text. Input should only be one test sample, not multiple.

        :param test_instance: The text string to analyze and predict the class (sentiment) of.
        :return: Returns the predicted class for the input text.
        '''
        
        # Transform the test documents into a DxV matrix, similar to that for the training documents
        # D is the number of test documents (in this case just 1 since analyzing document by document)
        # V is the number of types in the training documents.
        # Each cell in the matrix indicates the frequency (count) of a type in the document.
        test_count = self.count_vectorizer.transform([test_instance])
        
        # Predict the class for the test document
        return self.trained_lf.predict(test_count)[0]


class Lexicon:
    '''
    A lexicon-based sentiment classification baseline. Takes in a file of positive words and a file of negative words. Classify text based on which type of word it has more of.
    '''
    
    def __init__(self, pos_words, neg_words):
        '''
        Initializes the lexicon classification. Must be given the training data.

        :param pos_words: A file name containing words classified as positive (one word per line).
        :param neg_words: A file name containing words classified as negative (one word per line).
        '''
        
        self.posneg_words = {}
        self.train(pos_words, neg_words)

    
    def train(self, pos_words, neg_words):
        '''
        Makes a dictionary containing all of the training words. If a positive word, value is 1, if negative the value is -1.

        :param pos_words: A file name containing words classified as positive (one word per line).
        :param neg_words: A file name containing words classified as negative (one word per line).
        '''

        self.posneg_words.update({x.strip(): 1 for x in open(pos_words, encoding='utf8')})
        self.posneg_words.update({x.strip(): -1 for x in open(neg_words, encoding='utf8')})

    
    def classify(self, test_instance):
        '''
        Classifies text as positive, negative, or neutral based on the training words.
        If the text has more positive words, it's positive, and same with negative words. If equal amount (or neither positive or negative words), neutral.

        :param test_instance: The text to analyze and classify.
        :return: Returns the classification given to the input text.
        '''
        
        # Breaks the input text into the separate words.
        tokenized_instance = tokenize(test_instance)

        # For each token in the instance,
        # if pos word +1 to value, if neg word -1 to word, if neutral word just +0 to value
        value = 0
        for token in tokenized_instance:
            value += self.posneg_words.get(token, 0)
        if value > 0:
            return "positive"
        elif value < 0:
            return "negative"
        else:
            return "neutral"


class NaiveBayes:
    '''
    A Naive Bayes (with smoothing) multinomial sentiment classification model  
    '''
    def __init__(self, texts, classes):
        '''
        Initializes the Naive Bayes classification. Must be given the training data as it automatically trains when initialized.
        '''
        self.word_pos_freq, self.word_neg_freq, self.word_neu_freq = Counter(), Counter(), Counter() 

        self.word_pos_prob, self.word_neg_prob, self.word_neu_prob = {}, {}, {}

        self.unique_tokens = set()

        self.class_prob = {}
        self.train(texts, classes)

    
    def train(self, train_texts, train_classes):
        '''
        Goes through all the training documents and calculates the probability for each class of document.
        Also goes through each token seen in the documents and calculates the probability of the classes for each token.

        :param train_texts: A list of all of the texts being trained on.
        :param train_classes: A list of the class for each training sample.
        '''
        
        # Calculates the prob of a doc being a class
        # log(# docs in class k / # docs total)
        # Logarithm of result for less small calculations that could have underflow issues
        
        class_freqs = Counter(train_classes)
        for clas in class_freqs:
            self.class_prob[clas] = math.log(class_freqs[clas] / class_freqs.total())

        # Count the number of times a token appears in each class of document
        # Also store each unique token
        
        for index, text in enumerate(train_texts):
            tokenized_doc = tokenize(text)
            doc_type = train_classes[index]
            for token in tokenized_doc:
                if token not in self.unique_tokens:
                    self.unique_tokens.add(token)
                if doc_type == "positive":
                    self.word_pos_freq[token] += 1
                elif doc_type == "negative":
                    self.word_neg_freq[token] += 1
                elif doc_type == "neutral":
                    self.word_neu_freq[token] += 1

        # Go through all the unique tokens and calculate their prob (with Laplace add 1 smoothing) of appearing for each class
        # log(Freq token appeared in training docs of class c + 1 / (total tokens in docs of class c + # unique tokens))
        # Logarithm of result for less small calculations that could have underflow issues
        
        for token in self.unique_tokens:
            self.word_pos_prob[token] = math.log((self.word_pos_freq[token] + 1) /
                                                 (self.word_pos_freq.total() + len(self.unique_tokens)))
            self.word_neg_prob[token] = math.log((self.word_neg_freq[token] + 1) /
                                                 (self.word_neg_freq.total() + len(self.unique_tokens)))
            self.word_neu_prob[token] = math.log((self.word_neu_freq[token] + 1) /
                                                 (self.word_neu_freq.total() + len(self.unique_tokens)))
        

    def classify(self, test_instance):
        '''
        Takes some unclassified text and classifies the sentiment of it as either positive, negative, or neutral based on the training data.

        :param test_instance: The text to classify.
        :return: Returns the predicted class for the input text.
        '''
        
        # If for some reason a class doesn't appear in the training, it will never be used in classification
        class_options = {"positive": self.class_prob.get("positive", -math.inf),
                         "negative": self.class_prob.get("negative", -math.inf),
                         "neutral": self.class_prob.get("neutral", -math.inf)}

        for token in tokenize(test_instance):
            # Sum of the test doc's token probs + prior prob of class for all possible classes
            # If a token doesn't appear in the training documents, the default prob is 1 (since smoothing) / tokens in docs of class c + # unique tokens
            # and shift the result into log space
            class_options["positive"] += self.word_pos_prob.get(token, math.log(1 /
                                                 (self.word_pos_freq.total() + len(self.unique_tokens))))
            class_options["negative"] += self.word_neg_prob.get(token, math.log(1 /
                                                 (self.word_neg_freq.total() + len(self.unique_tokens))))
            class_options["neutral"] += self.word_neu_prob.get(token, math.log(1 /
                                                 (self.word_neu_freq.total() + len(self.unique_tokens))))

        # Getting key with the max value
        return max(class_options, key=class_options.get)


class BinaryNaiveBayes:
    '''
    A modified version of the Naive Bayes (with smoothing) multinomial sentiment classification model.
    A token in a single document will only count once (but if it shows up in separate documents it gets counted each time).
    '''
    
    def __init__(self, texts, classes):
        '''
        Initializes the modified Naive Bayes classification. Must be given the training data as it automatically trains when initialized.
        '''
        
        self.word_pos_freq, self.word_neg_freq, self.word_neu_freq = Counter(), Counter(), Counter() 

        self.word_pos_prob, self.word_neg_prob, self.word_neu_prob = {}, {}, {}

        self.unique_tokens = set()

        self.class_prob = {}

        self.train(texts, classes)

    
    def train(self, train_texts, train_classes):
        '''
        Goes through all the training documents and calculates the probability for each class of document.
        Also goes through each token seen in the documents and calculates the probability of the classes for each token.

        :param train_texts: A list of all of the texts being trained on.
        :param train_classes: A list of the class for each training sample.
        '''

        # Calculates the prob of a doc being a class
        # log(# docs in class k / # docs total)
        # Logarithm of result for less small calculations that could have underflow issues
        
        class_freqs = Counter(train_classes)
        for clas in class_freqs:
            self.class_prob[clas] = math.log(class_freqs[clas] / class_freqs.total())

        for index, text in enumerate(train_texts):
            tokenized_doc = tokenize(text)

            doc_set = set()
            doc_type = train_classes[index]

            for token in tokenized_doc:
                if token not in self.unique_tokens:
                    self.unique_tokens.add(token)
                
                # Only add token to count if not seen in the document previously
                if token not in doc_set:
                    doc_set.add(token)

                    if doc_type == "positive":
                        self.word_pos_freq[token] += 1

                    elif doc_type == "negative":
                        self.word_neg_freq[token] += 1

                    elif doc_type == "neutral":
                        self.word_neu_freq[token] += 1

        # Go through all the unique tokens and calculate their prob (with Laplace add 1 smoothing) of appearing for each class
        # log(Freq token appeared in training docs of class c + 1 / (total tokens in docs of class c + # unique tokens))
        # Logarithm of result for less small calculations that could have underflow issues
        
        for token in self.unique_tokens:
            self.word_pos_prob[token] = math.log((self.word_pos_freq[token] + 1) /
                                                 (self.word_pos_freq.total() + len(self.unique_tokens)))
            self.word_neg_prob[token] = math.log((self.word_neg_freq[token] + 1) /
                                                 (self.word_neg_freq.total() + len(self.unique_tokens)))
            self.word_neu_prob[token] = math.log((self.word_neu_freq[token] + 1) /
                                                 (self.word_neu_freq.total() + len(self.unique_tokens)))
    
    
    def classify(self, test_instance):
        '''
        Takes some unclassified text and classifies the sentiment of it as either positive, negative, or neutral based on the training data.

        :param test_instance: The text to classify.
        :return: Returns the predicted class for the input text.
        '''
        
        # If for some reason a class doesn't appear in the training, it will never be used in classification
        class_options = {"positive": self.class_prob.get("positive", -math.inf),
                         "negative": self.class_prob.get("negative", -math.inf),
                         "neutral": self.class_prob.get("neutral", -math.inf)}

        test_instance_unique = set()

        for token in tokenize(test_instance):
            # Each unique token is only used once in the calculation
            if token not in test_instance_unique:
                test_instance_unique.add(token)

                # Sum of the test doc's token probs + prior prob of class for all possible classes
                # If a token doesn't appear in the training documents, the default prob is 1 (since smoothing) / tokens in docs of class c + # unique tokens
                # and shift the result into log space
                class_options["positive"] += self.word_pos_prob.get(token, math.log(1 /
                                                    (self.word_pos_freq.total() + len(self.unique_tokens))))
                class_options["negative"] += self.word_neg_prob.get(token, math.log(1 /
                                                    (self.word_neg_freq.total() + len(self.unique_tokens))))
                class_options["neutral"] += self.word_neu_prob.get(token, math.log(1 /
                                                 (self.word_neu_freq.total() + len(self.unique_tokens))))

        # Getting key with the max value
        return max(class_options, key=class_options.get)
