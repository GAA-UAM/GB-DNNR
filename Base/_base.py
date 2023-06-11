""" Gradient Boosted - Deep Neural Network - Multi Output """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)
import os
import gc
import glob
import keras
import numpy as np
import tensorflow as tf
from Base._params import Params
from tensorflow.keras import regularizers

from abc import abstractmethod


class BaseEstimator(Params):
    def __init__(
        self,
        iter=50,
        eta=0.1,
        learning_rate=1e-3,
        total_nn=200,
        num_nn_step=1,
        batch_size=128,
        early_stopping=10,
        random_state=None,
        l2=0.01,
        dropout=0.1,
        record=False,
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
        self.record = record

    @abstractmethod
    def _validate_y(self, y):
        """validate y and specify the loss function"""

    def _layer_freezing(self, model):
        name = model.layers[-2].name
        model.get_layer(name).trainable = False
        self.layers.append(model.get_layer(name))
        assert (
            model.get_layer(name).trainable == False
        ), "The intermediate layer is not frozen!"

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
        model.add(
            keras.layers.Dense(
                self.num_nn_step,
                input_dim=X.shape[1],
                activation="relu",
                kernel_regularizer=regularizers.l2(self.l2),
            )
        )

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
        layer = keras.layers.Dense(
            self.num_nn_step,
            activation="relu",
            kernel_regularizer=regularizers.l2(self.l2),
        )
        layer.trainable = True
        model.add(layer)

        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(self.dropout))

        # Output layers
        model.add(keras.layers.Dense(self.n_classes))

        assert model.trainable == True, "Check the model trainability"
        assert (
            model.layers[-2].trainable == True
        ), "The new hidden layer should be trainable."

        return model

    def fit(self, X, y, x_test=None, y_test=None):
        X = X.astype(np.float32)

        y = self._validate_y(y)
        self._check_params()
        self._lists_initialization()

        if self.record:
            self._validate_y(y_test)
            val_data = (x_test, y_test)
            acum_test = np.ones_like(y_test) * self._loss.model0(y_test)
        else:
            val_data = (X, y)

        T = int(self.total_nn / self.num_nn_step)
        epochs = self.iter

        self.intercept = self._loss.model0(y)
        acum = np.ones_like(y) * self.intercept

        patience = self.iter if not self.early_stopping else self.early_stopping
        es = keras.callbacks.EarlyStopping(
            monitor="mean_squared_error", patience=patience, verbose=0
        )

        for i in range(T):
            residuals = self._loss.derive(y, acum)
            residuals = residuals.astype(np.float32)

            model = self._regressor(X=X, name=str(i))

            opt = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
                name="Adam",
            )

            model.compile(
                loss="mean_squared_error",
                optimizer=opt,
                metrics=[tf.keras.metrics.MeanSquaredError()],
            )

            self.history = model.fit(
                X,
                residuals,
                batch_size=self.batch_size,
                epochs=epochs,
                callbacks=[es],
                validation_data=val_data
            )

            self._layer_freezing(model=model)

            pred = model.predict(X)
            rho = self.eta * 1
            acum = acum + rho * pred
            self._add(model, rho)

            if self.record:
                pred_test = model.predict(x_test)
                acum_test = acum_test + rho * pred_test
                self.g_history["acc_val"].append(self._in_train_score(x_test, y_test))
                self.g_history["acc_train"].append(self._in_train_score(X, y))
                self.g_history["loss_train"].append(np.mean(self._loss(y, acum)))
                self.g_history["loss_test"].append(
                    np.mean(self._loss(y_test, acum_test))
                )
                self._save_records(epoch=i)

    def decision_function(self, X):
        pred = self._models[0].predict(X)
        raw_predictions = pred * self.steps[0] + self.intercept
        self._pred = raw_predictions

        for model, step in zip(self._models[1:], self.steps[1:]):
            raw_predictions += model.predict(X) * step

        return raw_predictions

    def _in_train_score(self, X, y):
        def rmse(y, pred):
            output_errors = np.sqrt(np.average((y - pred) ** 2, axis=0))
            return np.average(output_errors)

        pred = self.decision_function(X)
        return rmse(y, pred)

    def _save_records(self, epoch):
        def _path(archive):
            path = os.path.join("records", archive)
            return path

        archives = [
            ("loss_train.txt", self.g_history["loss_train"]),
            ("loss_test.txt", self.g_history["loss_test"]),
            ("rmse_train.txt", self.g_history["acc_train"]),
            ("rmse_val.txt", self.g_history["acc_val"]),
            (
                f"epoch_{str(epoch)}_additive_training_loss.txt",
                self.history.history["loss"],
            ),
            (
                f"epoch_{str(epoch)}_additive_val_loss.txt",
                self.history.history["val_loss"],
            ),
        ]
        for archive in archives:
            np.savetxt(_path(archive[0]), archive[1])

    def _check_params(self):
        """Check validity of parameters."""

        tf.keras.backend.clear_session()

        if self.record:
            try:
                os.mkdir("records")
            except:
                files = glob.glob("*/records/*")
                for f in files:
                    os.remove(f)

        if self.total_nn < self.num_nn_step:
            raise ValueError(
                f"Boosting number {self.total_nn} should be greater than the units {self.num_nn_step}."
            )

        if self.random_state is None:
            raise ValueError("Expected `seed` argument to be an integer")
        else:
            tf.random.set_seed(self.random_state)
            np.random.RandomState(self.random_state)

    def _lists_initialization(self):
        self.g_history = {
            "loss_train": [],
            "loss_test": [],
            "acc_train": [],
            "acc_val": [],
        }

        self.layers = []
        self._models = []
        self.steps = []

    @abstractmethod
    def predict_stage(self, X):
        """Return the predicted value of each boosting iteration"""

    @abstractmethod
    def score(self, X, y):
        """Return the score (accuracy for classification and aRMSE for regression)"""
