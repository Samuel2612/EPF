from sklearn.linear_model import LogisticRegression

class MyLogisticRegression:
    def __init__(self, penalty='l1', solver='liblinear', max_iter=1000):
        """
        A wrapper for Logistic Regression with specified defaults.
        """
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.model = LogisticRegression(
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter
        )

    def fit(self, X, y):
        """
        Fit the logistic regression model to data.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict binary labels for the input data.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities for each class (for binary or multi-class).
        """
        return self.model.predict_proba(X)

    @property
    def coef_(self):
        """
        Returns the coefficient array of shape (n_features,).
        For a binary classification, sklearn returns shape (1, n_features) 
        internally, so we extract the first row.
        """
        return self.model.coef_[0]


    
