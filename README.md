Sentiment Analysis – Linear Classifiers



Overview



This repository implements three linear classifiers to perform sentiment analysis on Amazon food reviews using a unigram Bag-of-Words (BoW) representation.



Perceptron



Average Perceptron



Pegasos (SVM via stochastic sub-gradient with L2 regularization)



The project covers feature extraction (binary and counts), optional stop-word removal, validation hyperparameter tuning, final test evaluation, and explainability via top unigrams.



Objective



The goal is to build a compact, reproducible baseline for text classification and compare linear learners under consistent features. The pipeline demonstrates:



Feature engineering trade-offs (binary vs counts, with/without stop-words)



Practical tuning of T (epochs) and λ (Pegasos regularization)



Clear explainability using weight-based top words



Data and Techniques Used



Data: Pre-split TSV files

reviews\_train.tsv, reviews\_val.tsv, reviews\_test.tsv



Features: Unigram BoW from training only, with binary or counts vectors



Stop-words: Optional removal via stopwords.txt



Algorithms: Perceptron, Average Perceptron, Pegasos (η = 1/√t, L2)



Decision rule: (θ · x + θ₀) > 0 ⇒ +1, else −1



Repository Layout and Components



sentiment\_analysis/main.py — Problem 7 accuracies, Problem 8 tuning, final test evaluation, top unigrams



sentiment\_analysis/project1.py — algorithms and feature extraction (BoW dictionary, binary and counts, stop-word toggle)



sentiment\_analysis/utils.py — data loading, tuning helpers, plotting



sentiment\_analysis/reviews\_\*.tsv, toy\_data.tsv, stopwords.txt



images/ — screenshots referenced below (add your PNGs here)



README.md



Flow and Visuals



Row 1 (Sanity \& Tests)



All implemented functions verified locally





Row 2 (Model Intuition)



Toy 2D dataset with decision boundary (Perceptron example)





Row 3 (Explainability)



Top 10 most explanatory unigrams for the positive class (printed from main.py)





Results Summary



Validation (Problem 7; T = 10, λ = 0.01):



Perceptron: 0.7160



Average Perceptron: 0.7980



Pegasos: 0.7900



Best after tuning (Problem 8):



Perceptron: T = 25 → 0.7940



Average Perceptron: T = 25 → 0.8000



Pegasos: T = 25, λ = 0.01 → 0.8060



Final Test (Pegasos, T = 25, λ = 0.01):



Original dictionary (binary): 0.8020



Stop-words removed (binary): 0.8080



Stop-words removed (counts): 0.7700



Insights



Binary BoW outperforms counts for this dataset.



Stop-word removal slightly improves binary-feature accuracy.



Pegasos gives the strongest validation performance among the three.



Usage and Navigation



Environment



Activate conda env and install dependencies: numpy, matplotlib



Run tests



Navigate to sentiment\_analysis and run test.py



Run experiments



Execute main.py to print accuracies (Problem 7), run tuning (Problem 8), output final test accuracy, and list top unigrams



Feature variants



Use binary or counts features and toggle stop-words in project1.py (dictionary is always built from training only)



Technologies Used



Python, NumPy, matplotlib



Anaconda for environment management



Conclusion



Linear classifiers are strong baselines for text sentiment with BoW features.



Pegasos (with tuned T and λ) is robust and competitive.



Simple interpretability via top weights provides actionable insight into learned signals.

