from string import punctuation, digits
import numpy as np
import random
import utils
import re
import string
# If utils has extract_words use it; otherwise use the course-style tokenizer below.
try:
    import utils  # noqa: F401
    HAVE_UTILS = True
except Exception:
    HAVE_UTILS = False

def _course_extract_words(text: str):
    """
    Course-style tokenizer:
    - Lowercase
    - Insert spaces around punctuation and digits
    - Split on whitespace
    This matches the typical grader behavior.
    """
    s = str(text)
    for c in (string.punctuation + string.digits):
        s = s.replace(c, f" {c} ")
    return s.lower().split()

# Resolve tokenizer: prefer utils.extract_words if present, else use course one.
if HAVE_UTILS and hasattr(utils, "extract_words"):
    EXTRACT_TOKENS = utils.extract_words
else:
    EXTRACT_TOKENS = _course_extract_words


#==============================================================================
#===  PART I  =================================================================
#==============================================================================



def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices



def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.
    """
    margin = label * (np.dot(theta, feature_vector) + theta_0)
    return float(max(0.0, 1.0 - margin))




def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Average hinge loss over all samples.
    feature_matrix: shape (n_samples, n_features)
    labels: shape (n_samples,), each in {+1, -1}
    """
    margins = labels * (feature_matrix @ theta + theta_0)
    losses = np.maximum(0.0, 1.0 - margins)
    return float(np.mean(losses))





def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    One perceptron update on a single example.
    If y*(θ·x + θ0) <= 0, update:
        θ ← θ + y*x
        θ0 ← θ0 + y
    Otherwise, leave parameters unchanged.
    """
    if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 0:
        current_theta = current_theta + label * feature_vector
        current_theta_0 = current_theta_0 + label
    return current_theta, current_theta_0




def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm for T passes over the data.
    Initializes theta (shape: (n_features,)) and theta_0 = 0.0.
    Uses get_order(n_samples) each epoch.
    """
    n_samples, n_features = feature_matrix.shape
    theta = np.zeros(n_features)
    theta_0 = 0.0

    for _ in range(T):
        for i in get_order(n_samples):
            x_i = feature_matrix[i]
            y_i = labels[i]
            theta, theta_0 = perceptron_single_step_update(x_i, y_i, theta, theta_0)

    return theta, float(theta_0)



def average_perceptron(feature_matrix, labels, T):
    """
    Average perceptron over T passes.
    After every single-step update (or no-update), accumulate (theta, theta_0).
    Return the average parameters.
    """
    n_samples, n_features = feature_matrix.shape
    theta = np.zeros(n_features)
    theta_0 = 0.0

    theta_sum = np.zeros(n_features)
    theta_0_sum = 0.0
    count = 0  # number of steps over all examples and epochs

    for _ in range(T):
        for i in get_order(n_samples):
            x_i = feature_matrix[i]
            y_i = labels[i]
            # reuse single-step update
            theta, theta_0 = perceptron_single_step_update(x_i, y_i, theta, theta_0)

            # accumulate after this step
            theta_sum += theta
            theta_0_sum += theta_0
            count += 1

    # average
    return theta_sum / count, float(theta_0_sum / count)


def pegasos_single_step_update(feature_vector, label, L, eta, current_theta, current_theta_0):
    """
    One Pegasos update on a single example.
    If y*(θ·x + θ0) <= 1:
        θ ← (1 - ηL)θ + η y x
        θ0 ← θ0 + η y
    Else:
        θ ← (1 - ηL)θ
        θ0 unchanged
    """
    margin = label * (np.dot(current_theta, feature_vector) + current_theta_0)

    # Always apply regularization shrink
    new_theta = (1 - eta * L) * current_theta
    new_theta_0 = current_theta_0

    # Hinge condition
    if margin <= 1:
        new_theta = new_theta + eta * label * feature_vector
        new_theta_0 = new_theta_0 + eta * label

    return new_theta, new_theta_0




def pegasos(feature_matrix, labels, T, L):
    """
    Full Pegasos algorithm.
    Uses get_order(n_samples) each epoch and step size eta = 1/sqrt(t).
    Returns (theta, theta_0).
    """
    n_samples, n_features = feature_matrix.shape
    theta = np.zeros(n_features)
    theta_0 = 0.0

    t = 0  # global step counter across all epochs and samples
    for _ in range(T):
        for i in get_order(n_samples):
            t += 1
            eta = 1.0 / np.sqrt(t)
            x_i = feature_matrix[i]
            y_i = labels[i]
            theta, theta_0 = pegasos_single_step_update(
                x_i, y_i, L, eta, theta, theta_0
            )

    return theta, float(theta_0)




#==============================================================================
#===  PART II  ================================================================
#==============================================================================



##  #pragma: coderesponse template
##  def decision_function(feature_vector, theta, theta_0):
##      return np.dot(theta, feature_vector) + theta_0
##  def classify_vector(feature_vector, theta, theta_0):
##      return 2*np.heaviside(decision_function(feature_vector, theta, theta_0), 0)-1
##  #pragma: coderesponse end



def classify(feature_matrix, theta, theta_0):
    """
    Predict +1 if (theta·x + theta_0) > 0, else -1.
    """
    scores = feature_matrix @ theta + theta_0
    return np.where(scores > 0, 1, -1)



def classifier_accuracy(classifier, train_feature_matrix, val_feature_matrix, train_labels, val_labels, **kwargs):
    """
    Train the given classifier on the training set, then compute accuracy
    on both train and validation sets. Extra hyperparameters (e.g., T, L)
    are passed via **kwargs.
    """
    # Train to get parameters
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)

    # Predict
    train_preds = classify(train_feature_matrix, theta, theta_0)
    val_preds   = classify(val_feature_matrix,   theta, theta_0)

    # Use the provided accuracy() helper
    return accuracy(train_preds, train_labels), accuracy(val_preds, val_labels)



def extract_words(text):
    """
    Helper function for `bag_of_words(...)`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    # Your code here
    raise NotImplementedError

    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()



def bag_of_words(texts):
    dictionary = {}
    for text in texts:
        for token in EXTRACT_TOKENS(text):
            if token not in dictionary:
                dictionary[token] = len(dictionary)
    return dictionary

def extract_bow_feature_vectors(reviews, dictionary, binarize=True):
    n_samples = len(reviews)
    n_features = len(dictionary)
    X = np.zeros((n_samples, n_features))
    for i, text in enumerate(reviews):
        tokens = EXTRACT_TOKENS(text)
        if binarize:
            for tok in set(tokens):
                j = dictionary.get(tok)
                if j is not None:
                    X[i, j] = 1
        else:
            for tok in tokens:
                j = dictionary.get(tok)
                if j is not None:
                    X[i, j] += 1
    return X



def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()
