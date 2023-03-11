""" Gradient Boosted - Deep Neural Network - Multi Output """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import gc
import keras
import numpy as np
import tensorflow as tf
from Base._params import Params
from tensorflow.keras import regularizers

from abc import abstractmethod


class BaseEstimator(Params):

    def __init__(self,
                 iter=50,
                 eta=0.1,
                 learning_rate=1e-3,
                 total_nn=200,
                 num_nn_step=1,
                 batch_size=128,
                 early_stopping=10,
                 random_state=None,
                 l2=0.01,
                 dropout=0.1
                 ):

        self.iter = iter
        self.eta = eta
        self.learning_rate = learning_rate
        self.total_nn = total_nn
        self.num_nn_step = num_nn_step
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.l2 = l2
        self.dropout = dropout

    @abstractmethod
    def _validate_y(self, y):
        """validate y and specify the loss function"""

    def _layer_freezing(self, model):
        name = model.layers[-2].name
        model.get_layer(name).trainable = False
        self.layers.append(model.get_layer(name))
        assert model.get_layer(
            name).trainable == False, "The intermediate layer is not frozen!"

    def _add(self, model, step):
        cloned_model = tf.keras.models.clone_model(model)
        cloned_model.set_weights(model.get_weights())
        self._models.append(cloned_model)
        self.steps.append(step)
        del cloned_model
        gc.collect()

    def _regressor(self, X, name):
        """Building the additive deep
        regressor of the gradient boosting"""

        model = keras.models.Sequential(name=name)

        # Normalizing the input
        # model.add(tf.keras.layers.LayerNormalization(axis=-1))

        # Build the Input Layer
        model.add(keras.layers.Dense(self.num_nn_step,
                                        input_dim=X.shape[1],
                                        activation="relu", 
                                        kernel_regularizer=regularizers.l2(self.l2)))
        
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(self.dropout))
        # Hidden Layers
        # Empowering the network with frozen trained layers
        for layer in self.layers:
            # Importing frozen layers as the intermediate layers of the network
            model.add(layer)
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(self.dropout))

        # Adds one new raw hidden layer with randomized weight
        # get_weights()[0].shape == (self.num_nn_step, self.num_nn_step)
        # get_weights()[1].shape == (self.num_nn_step)
        layer = keras.layers.Dense(self.num_nn_step,
                                   activation="relu", 
                                   kernel_regularizer=regularizers.l2(self.l2))
        layer.trainable = True
        model.add(layer)

        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(self.dropout))


        # Output layers
        model.add(keras.layers.Dense(self.n_classes))

        assert model.trainable == True, "Check the model trainability"
        assert model.layers[-2].trainable == True, "The new hidden layer should be trainable."

        return model

    def fit(self, X, y):

        X = X.astype(np.float32)

        y = self._validate_y(y)
        self._check_params()
        self._lists_initialization()

        T = int(self.total_nn/self.num_nn_step)
        epochs = self.iter

        self.intercept = self._loss.model0(y)
        acum = np.ones_like(y) * self.intercept

        patience = self.iter if not self.early_stopping else self.early_stopping
        es = keras.callbacks.EarlyStopping(monitor="mean_squared_error",
                                           patience=patience,
                                           verbose=0)

        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=1e-07,
                                       amsgrad=False,
                                       name="Adam")

        for i in (range(T)):

            residuals = self._loss.derive(y, acum)
            residuals = residuals.astype(np.float32)

            model = self._regressor(X=X,
                                    name=str(i)
                                    )

            model.compile(loss="mean_squared_error",
                          optimizer=opt,
                          metrics=[tf.keras.metrics.MeanSquaredError()])

            model.fit(X, residuals,
                      batch_size=self.batch_size,
                      epochs=epochs,
                      callbacks=[es],
                      )

            self._layer_freezing(model=model)

            pred = model.predict(X)
            rho = self.eta * 1
            acum = acum + rho * pred

            self._reg_score.append(model.evaluate(X,
                                                  residuals,
                                                  verbose=0)[1])
            self._loss_curve.append(np.mean(self._loss(y, acum)))
            self._add(model, rho)

    def decision_function(self, X):

        pred = self._models[0].predict(X)
        raw_predictions = pred * self.steps[0] + self.intercept
        self._pred = raw_predictions

        for model, step in zip(self._models[1:], self.steps[1:]):
            raw_predictions += model.predict(X) * step

        return raw_predictions

    def _check_params(self):
        """Check validity of parameters."""

        tf.keras.backend.clear_session()

        if self.total_nn < self.num_nn_step:
            raise ValueError(
                f"Boosting number {self.total_nn} should be greater than the units {self.num_nn_step}.")

        if self.random_state is None:
            raise ValueError("Expected `seed` argument to be an integer")
        else:
            tf.random.set_seed(self.random_state)
            np.random.RandomState(self.random_state)

    def _lists_initialization(self):
        self.layers = []
        self._reg_score = []
        self._loss_curve = []
        self._models = []
        self.steps = []

    @abstractmethod
    def predict_stage(self, X):
        """Return the predicted value of each boosting iteration"""

    @abstractmethod
    def score(self, X, y):
        """Return the score (accuracy for classification and aRMSE for regression)"""

