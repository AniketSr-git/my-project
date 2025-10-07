Sentiment Analysis – Linear Classifiers (Project 1)



Three linear classifiers on Amazon food reviews with unigram Bag-of-Words features.



Algorithms: Perceptron, Average Perceptron, Pegasos (SVM with ℓ2 regularization)



Features: Unigram bag-of-words (binary or counts), optional stop-word removal



Data: reviews\_train.tsv, reviews\_val.tsv, reviews\_test.tsv



Repository Structure



sentiment\_analysis/main.py — run accuracies (Problem 7), tuning (Problem 8), final eval, top unigrams



sentiment\_analysis/project1.py — algorithms + feature extraction (BoW, counts, stopwords toggle)



sentiment\_analysis/utils.py — data loading, tuning helpers, plotting



sentiment\_analysis/reviews\_\*.tsv, toy\_data.tsv, stopwords.txt



docs/img/ — screenshots (add your PNGs here)



README.md



What’s Implemented



Algorithms



Hinge loss (single \& full)



Perceptron: single-step update + full loop



Average Perceptron: parameter averaging



Pegasos: single-step + full (step size η = 1/√t, ℓ2 regularization)



Feature Engineering



Unigram Bag-of-Words built from training only



Binary features (presence/absence) and Counts features



Stop-words removal via stopwords.txt (toggle in project1.py)



Evaluation \& Tuning



classify() with strict boundary rule (> 0 ⇒ +1, else −1)



classifier\_accuracy() to train \& evaluate



Tuning (Problem 8)



Perceptron / Avg-Perceptron: T ∈ {1, 5, 10, 15, 25, 50}



Pegasos: tune T at fixed λ = 0.01, then tune λ ∈ {0.001, 0.01, 0.1, 1, 10}



Setup



Activate env and install deps:



conda activate myproj



cd C:\\Users\\aniks\\my-project



python -m pip install numpy matplotlib



How To Run



Run experiments:



cd sentiment\_analysis



python main.py



Key results (your values may vary slightly):



Problem 7 (T = 10, λ = 0.01) — validation



Perceptron: 0.7160



Average Perceptron: 0.7980



Pegasos: 0.7900



Problem 8 (tuning) — best validation



Perceptron: T = 25 → 0.7940



Average Perceptron: T = 25 → 0.8000



Pegasos: T = 25, λ = 0.01 → 0.8060



Final test (Pegasos, T=25, λ=0.01)



Original dictionary: 0.8020



Stop-words removed (binary): 0.8080



Stop-words removed (counts): 0.7700



Screenshots \& Plots



Add PNGs to docs/img/:



docs/img/tests\_passed.png — green tests summary



docs/img/perceptron\_plot.png — toy decision boundary



docs/img/top\_unigrams.png — optional table/screenshot



Embed examples (after you add files):



Design Notes



Why Pegasos? Decaying step + ℓ2 regularization gives stable convergence and best validation here.



Binary vs Counts: Counts can overweight very frequent terms; binary often works better with linear margins on this dataset.



Stop-words: Removing common function words reduces noise and slightly improves binary-feature accuracy.



Reproducibility Checklist



Build vocabulary only from training data



Keep tokenizer consistent (course style: space around punctuation/digits, lowercase)



Use the same tuning grids as in main.py



Use strict boundary in classify (> 0 ⇒ +1, else −1)



Quick Commands



Run tests: cd sentiment\_analysis \&\& python test.py



Run main: cd sentiment\_analysis \&\& python main.py



Add screenshots and push:



mkdir docs \&\& mkdir docs\\img



copy PNGs into docs\\img



git add docs\\img\\\*.png



git commit -m "docs: add tests/plot screenshots"



git push



Close Environment \& Detach Git



Close Anaconda session:



conda deactivate



exit



Detach this repo from remote (optional):



cd C:\\Users\\aniks\\my-project



git remote -v



git remote remove origin



git remote -v (should be empty)



Clear cached GitHub credentials (Windows):



Control Panel → Credential Manager → Windows Credentials → remove github.com / git:https://github.com

