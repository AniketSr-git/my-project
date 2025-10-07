import project1 as p1
import utils
import numpy as np

# ------------------------------------------------------------------------------
# Data loading. There is no need to edit code in this section.
# ------------------------------------------------------------------------------
train_data = utils.load_data("reviews_train.tsv")
val_data   = utils.load_data("reviews_val.tsv")
test_data  = utils.load_data("reviews_test.tsv")

train_texts, train_labels = zip(*((s["text"], s["sentiment"]) for s in train_data))
val_texts,   val_labels   = zip(*((s["text"], s["sentiment"]) for s in val_data))
test_texts,  test_labels  = zip(*((s["text"], s["sentiment"]) for s in test_data))

# ------------------------------------------------------------------------------
# Bag-of-words (binary features) built from TRAIN ONLY
# ------------------------------------------------------------------------------
dictionary         = p1.bag_of_words(train_texts)
train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
val_bow_features   = p1.extract_bow_feature_vectors(val_texts, dictionary)
test_bow_features  = p1.extract_bow_feature_vectors(test_texts, dictionary)

# ------------------------------------------------------------------------------
# Problem 7 — Print accuracies for T=10, L=0.01
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    T = 10
    L = 0.01

    pct_train_acc, pct_val_acc = p1.classifier_accuracy(
        p1.perceptron,
        train_bow_features, val_bow_features,
        train_labels, val_labels,
        T=T
    )
    print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_acc))
    print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_acc))

    avg_train_acc, avg_val_acc = p1.classifier_accuracy(
        p1.average_perceptron,
        train_bow_features, val_bow_features,
        train_labels, val_labels,
        T=T
    )
    print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_train_acc))
    print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_val_acc))

    peg_train_acc, peg_val_acc = p1.classifier_accuracy(
        p1.pegasos,
        train_bow_features, val_bow_features,
        train_labels, val_labels,
        T=T, L=L
    )
    print("{:50} {:.4f}".format("Training accuracy for Pegasos:", peg_train_acc))
    print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", peg_val_acc))

# ------------------------------------------------------------------------------
# Problem 8 — Hyperparameter tuning
#   Ts = [1, 5, 10, 15, 25, 50]
#   Ls = [0.001, 0.01, 0.1, 1, 10]
#   Pegasos: first fix L=0.01 to tune T; then fix best T to tune L.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    data = (train_bow_features, train_labels, val_bow_features, val_labels)
    Ts = [1, 5, 10, 15, 25, 50]
    Ls = [0.001, 0.01, 0.1, 1, 10]

    # Perceptron tuning over T
    pct_tune_results = utils.tune_perceptron(Ts, *data)
    print("perceptron valid:", list(zip(Ts, pct_tune_results[1])))
    print("best = {:.4f}, T={}".format(
        float(np.max(pct_tune_results[1])),
        Ts[int(np.argmax(pct_tune_results[1]))]
    ))

    # Average Perceptron tuning over T
    avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
    print("avg perceptron valid:", list(zip(Ts, avg_pct_tune_results[1])))
    print("best = {:.4f}, T={}".format(
        float(np.max(avg_pct_tune_results[1])),
        Ts[int(np.argmax(avg_pct_tune_results[1]))]
    ))

    # Pegasos: first fix lambda=0.01 and tune T
    fix_L = 0.01
    peg_tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
    print("Pegasos valid: tune T", list(zip(Ts, peg_tune_results_T[1])))
    print("best = {:.4f}, T={}".format(
        float(np.max(peg_tune_results_T[1])),
        Ts[int(np.argmax(peg_tune_results_T[1]))]
    ))

    # Then fix T to that best and tune lambda
    fix_T = Ts[int(np.argmax(peg_tune_results_T[1]))]
    peg_tune_results_L = utils.tune_pegasos_L(fix_T, Ls, *data)
    print("Pegasos valid: tune L", list(zip(Ls, peg_tune_results_L[1])))
    print("best = {:.4f}, L={}".format(
        float(np.max(peg_tune_results_L[1])),
        Ls[int(np.argmax(peg_tune_results_L[1]))]
    ))

# ------------------------------------------------------------------------------
# Final test evaluation with best model/params from tuning
# (Fill with the best you observed to avoid re-tuning every run)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    BEST_ALGO = "pegasos"
    BEST_T = 25
    BEST_L = 0.01

    if BEST_ALGO == "perceptron":
        _, val_acc = p1.classifier_accuracy(
            p1.perceptron, train_bow_features, val_bow_features,
            train_labels, val_labels, T=BEST_T
        )
        theta, theta0 = p1.perceptron(train_bow_features, train_labels, T=BEST_T)

    elif BEST_ALGO == "avg_perceptron":
        _, val_acc = p1.classifier_accuracy(
            p1.average_perceptron, train_bow_features, val_bow_features,
            train_labels, val_labels, T=BEST_T
        )
        theta, theta0 = p1.average_perceptron(train_bow_features, train_labels, T=BEST_T)

    else:  # Pegasos
        _, val_acc = p1.classifier_accuracy(
            p1.pegasos, train_bow_features, val_bow_features,
            train_labels, val_labels, T=BEST_T, L=BEST_L
        )
        theta, theta0 = p1.pegasos(train_bow_features, train_labels, T=BEST_T, L=BEST_L)

    test_preds = p1.classify(test_bow_features, theta, theta0)
    test_acc = float(np.mean(test_preds == np.array(test_labels)))
    print("Chosen model:", BEST_ALGO, "T=", BEST_T, "L=", BEST_L)
    print("Validation accuracy (recomputed):", round(float(val_acc), 4))
    print("Test accuracy:", round(test_acc, 4))
