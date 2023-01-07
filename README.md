# Car-Price-Regression-and-Classification
Pattern Recognition Kaggle Competition

# Problem Definition:-
based on the provided features in the updated dataset first, Predict a Car Price 
then providing a Categorical Target, Classify a car price into one of four categories: {cheap, moderate, expensive or very expensive}

Note: You must preprocess all features, but the model and feature selection can be done
after that (i.e You can drop a feature only after preprocessing and with a valid reason)

# Workflow:-
- Applied Data Cleaning separate combined car info to its elements model, manufacture and Year, also normalizing same categories but have different string representations.
- Keeping Statistical Metrics precomputed on Training set and using it on Testing set during performing Standardization for Numerical Values.
- Did One-Hot Encoding for Categorical Values and handle the unknow category comes from Testing by a new Placeholder Category named ‘unknown’
- Trained Polynomial Regression with degree=3, got 0.89 best Accuracy
- Trained 3 Model SVM, Decision Tree, as well as Ensemble Method Random Forrest, got .89 best Accuracy

# Testing:- [Incompleted yet]
