import os
import numpy as np
import random
from collections import Counter
import copy
import matplotlib.pyplot as plt
import pandas as pd
import pandasql as ps

def folder_list(path,label):
    '''
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    '''
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r.append(label)
        review.append(r)
    return review

def read_data(file):
    '''
    Read each file into a list of strings.
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on',
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    f = open(file)
    lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = map(lambda Element: Element.translate(str.maketrans("", "", symbols)).strip(), lines)
    words = filter(None, words)
    return list(words)


def load_and_shuffle_data():
    '''
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    '''
    pos_path = "/Users/levpaciorkowski/Documents/Spring_2022/ML/hw3/data/pos"
    neg_path = "/Users/levpaciorkowski/Documents/Spring_2022/ML/hw3/data/neg"

    pos_review = folder_list(pos_path,1)
    neg_review = folder_list(neg_path,-1)

    review = pos_review + neg_review
    random.shuffle(review)
    return review

# Taken from http://web.stanford.edu/class/cs221/ Assignment #2 Support Code
def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale


### Q6
def bag_of_words_representation(text):
    """
    Parameters
    ----------
    text : list of str

    Returns
    -------
    dictionary representation of text
    """
    out = dict()
    for word in text:
        if word in out.keys():
            out[word] += 1
        else:
            out[word] = 1
    return out



### Q7
def init_data():
    pos = "/Users/levpaciorkowski/Documents/Spring_2022/ML/hw3/data/pos"
    neg = "/Users/levpaciorkowski/Documents/Spring_2022/ML/hw3/data/neg"
    positive = folder_list(pos, 1)
    negative = folder_list(neg, -1)
    data = []

    # convert each example into its sparse dictionary representation
    for i in range(len(positive)):
        review = positive[i]
        label = review.pop()
        bow = bag_of_words_representation(review)
        data.append((bow, label))
        
        review = negative[i]
        label = review.pop()
        bow = bag_of_words_representation(review)
        data.append((bow, label))
    
    random.shuffle(data)
        
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(len(data)):
        if i < 1500:
            X_train.append(data[i][0])
            y_train.append(data[i][1])
        else:
            X_test.append(data[i][0])
            y_test.append(data[i][1])
    
    return X_train, y_train, X_test, y_test



### Q8
def pegasos(lbda, X_train, y_train, epochs):
    # initialize w, the weight vector and t, iteration number
    # training data has already been randomly shuffled
    w = dict() # empty dict
    t = 0
    epoch = 0
    while epoch < epochs:
        for j in range(len(X_train)):
            t += 1
            if t % 100 == 0:
                print("Step number: " + str(t))
            nu = 1/(lbda*t)
            y_j = copy.deepcopy(y_train[j])
            x_j = copy.deepcopy(X_train[j])
            # now compute the margin, y_j@w.T@x_j
            m = y_j*dotProduct(w, x_j)
            increment(w, -nu*lbda, w)
            if m < 1:
                increment(w, nu*y_j, x_j)
        epoch += 1
    return w


### Q9
def fast_pegasos(lbda, X_train, y_train, epochs):
    # pegasos algorithm with more efficient implementation of w
    s = 1
    W = dict()
    # w = sW, where s is the scaling factor
    t = 1
    epoch = 0
    while epoch < epochs:
        for j in range(len(X_train)):
            t += 1
            if t % 100 == 0:
                print("Step number: " + str(t))
            nu = 1/(lbda*t)
            s = (1 - nu*lbda) * s
            y_j = copy.deepcopy(y_train[j])
            x_j = copy.deepcopy(X_train[j])
            m = y_j*dotProduct(W, x_j) * s
            if m < 1:
                increment(W, nu*y_j/s, x_j)
        epoch += 1
    for key in W:
        W[key] *= s
    return W
        
                
    
def classification_error(w, X_test, y_test):
    correct, n = 0, 0
    for i in range(len(X_test)):
        test_sample = X_test[i]
        test_label = y_test[i]
        prediction = dotProduct(w, test_sample)
        m = test_label * prediction
        n += 1
        if m > 0:
            correct += 1
    return 1 - correct/n




X_train, y_train, X_test, y_test = init_data()
w = pegasos(0.01, X_train, y_train, 10)
# takes about 59 sec for lbda = 0.01, epochs = 10; error = 0.46

w = fast_pegasos(0.01, X_train, y_train, 10)
# takes about 5 sec for lbda = 0.01, epochs = 10; error = 0.16

lbda_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

lbdas = dict.fromkeys(lbda_list)

for lbda in lbda_log_list:
    print("Trying labmda = " + str(lbda))
    w = fast_pegasos(lbda, X_train, y_train, 100)
    error = classification_error(w, X_test, y_test)
    lbdas[lbda] = error
    

horizontal = list(lbdas.keys())
for i in range(len(horizontal)):
    horizontal[i] = np.log10(horizontal[i])
    
vertical = list(lbdas.values())

plt.plot(horizontal, vertical)
plt.xlabel("log10(lambda)")
plt.ylabel("classification error")
plt.title("Classification Error for different values of Lambda")


error = classification_error(w_optimal, X_test, y_test)

lbda_log_list = [-4.3, -4.2, -4.1, -4, -3.9, -3.8, -3.7, -1.3, -1.2, -1.1, -1, -0.9, -0.8, -0.7]

for i in range(len(lbda_log_list)):
    lbda_log_list[i] = 10**lbda_log_list[i]


data = sorted(lbdas.items())
horizontal, vertical = [], []
for i in data:
    horizontal.append(np.log10(i[0]))
    vertical.append(i[1])


optimal_lambda = min(lbdas, key=lbdas.get)

w_optimal = fast_pegasos(optimal_lambda, X_train, y_train, 100)




### Q13
scores = []
correct = []
for i in range(len(y_test)):
    score = dotProduct(w_optimal, X_test[i])
    scores.append(score)
    m = score * y_test[i]
    correct.append(1) if m > 0 else correct.append(0)
    
q13 = pd.DataFrame({'score': scores,
                   'correct': correct})

q13 = q13.sort_values('score')

score_top_bound = []
accuracy = []

quintile = []

q13['quintile'] = None
q13 = q13.reset_index()
for index, row in q13.iterrows():
    if row['magnitude'] <= score_top_bound[0]:
        quintile.append(1)
    elif row['magnitude'] <= score_top_bound[1]:
        quintile.append(2)
    elif row['magnitude'] <= score_top_bound[2]:
        quintile.append(3)
    elif row['magnitude'] <= score_top_bound[3]:
        quintile.append(4)
    else:
        quintile.append(5)

q13['quintile'] = quintile

q13['magnitude'] = abs(q13['score'])

scores = q13['magnitude']

qs = [0.2, 0.4, 0.6, 0.8, 1]

quantiles = scores.quantile([0.2, 0.4, 0.6, 0.8, 1])

score_top_bound = quantiles['magnitude']

for q in qs:
    score_top_bound.append(quantiles[q])



query = """SELECT quintile, avg(magnitude) AS avg_magnitude, sum(correct) AS percent_accuracy FROM q13 GROUP BY quintile"""

print(ps.sqldf(query, locals()))

### Q14
# index 341 was one prominent example the model classified incorrectly
# index 321 also the model got wrong
# index 34

i = 34

example = X_test[i]
label = y_test[i]
score = dotProduct(w_optimal, example)


feature_name = []
feature_value_xi = []
feature_weight_wi = []
importance_wixi = []

for feature in example.keys():
    if feature not in w_optimal.keys():
        continue
    feature_name.append(feature)
    xi = example[feature]
    feature_value_xi.append(xi)
    wi = w_optimal[feature]
    feature_weight_wi.append(wi)
    importance_wixi.append(abs(wi*xi))

i34 = pd.DataFrame({'feature_name': feature_name,
                     'feature_value_xi': feature_value_xi,
                     'feature_weight_wi': feature_weight_wi,
                     'importance_wixi': importance_wixi
                     })

i34 = i34.sort_values('importance_wixi', ascending=False)

i34.head(15)

false_positive = []
false_negative = []
for index, row in q13.iterrows():
    if row['correct'] == 0:
        if row['score'] > 0:
            false_positive.append(1)
            false_negative.append(0)
        else:
            false_negative.append(1)
            false_positive.append(0)
    else:
        false_positive.append(0)
        false_negative.append(0)

q13['false_positive'] = false_positive
q13['false_negative'] = false_negative


query = """SELECT quintile, sum(false_positive) AS false_positives, sum(false_negative) AS false_negatives FROM q13 GROUP BY quintile"""

print(ps.sqldf(query, locals()))













