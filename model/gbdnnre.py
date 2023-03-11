import numpy as np
from Base._base import BaseEstimator
from Base._loss import squared_loss

class DeepRegressor(BaseEstimator):

    def __init__(self,
                 iter=10,
                 eta=0.1,
                 learning_rate=1e-3,
                 total_nn=200,
                 num_nn_step=1,
                 batch_size=128,
                 early_stopping=False,
                 random_state=None,
                 l2 = 0.01,
                 dropuout=0.1):

        super().__init__(iter,
                         eta,
                         learning_rate,
                         total_nn,
                         num_nn_step,
                         batch_size,
                         early_stopping,
                         random_state,
                         l2,
                         dropuout)

    def _validate_y(self, y):
        self._loss = squared_loss()
        self.n_classes = y.shape[1] if len(y.shape) == 2 else 1
        return y

    def predict(self, X):
        return self.decision_function(X)

    def predict_stage(self, X):
        preds = np.ones_like(self._models[0].predict(X))*self.intercept

        for model, step in zip(self._models, self.steps):
            preds += model.predict(X) * step
            yield preds

    def score(self, X, y):
        """Returns the average of RMSE of all outputs."""
        pred = self.predict(X)
        output_errors = np.mean((y - pred) ** 2, axis=0)

        return np.mean(np.sqrt(output_errors))
