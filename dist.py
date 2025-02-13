import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers

import optuna
from sklearn.model_selection import train_test_split


def johnson_su_nll_loss(y_true, y_pred):
    """
    y_pred => (batch_size,4): raw parameters [raw_loc, raw_scale, raw_skew, raw_tail].
    
    We transform them for the JohnsonSU distribution:
      loc = raw_loc
      scale = softplus(raw_scale)   >0
      skewness = raw_skew          any real
      tailweight= softplus(raw_tail) >0
    Then compute -mean(log_prob).
    """
    raw_loc   = y_pred[:, 0]
    raw_scale = y_pred[:, 1]
    raw_skew  = y_pred[:, 2]
    raw_tail  = y_pred[:, 3]

    loc       = raw_loc
    scale     = tf.nn.softplus(raw_scale)
    skewness  = raw_skew
    tailweight= tf.nn.softplus(raw_tail)

    dist = tfd.JohnsonSU(loc=loc, scale=scale, skewness=skewness, tailweight=tailweight)
    nll = -tf.reduce_mean(dist.log_prob(y_true))
    return nll


def build_johnson_su_model(n_features, hidden_units=16, l1_strength=0.0, learning_rate=1e-3):
    """
    A simple feedforward net:
      - One hidden layer (size=hidden_units) with optional L1,
      - Outputs 4 "raw" parameters for the JohnsonSU distribution.
    """
    from tensorflow.keras import regularizers

    reg = regularizers.L1(l1_strength) if (l1_strength > 0) else None

    inputs = tfkl.Input(shape=(n_features,), name="Features")
    x = tfkl.Dense(hidden_units, activation='relu', kernel_regularizer=reg)(inputs)
    # Optionally add more layers, dropout, etc. here

    outputs = tfkl.Dense(4, activation=None, name="RawParams")(x)

    model = tfk.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=johnson_su_nll_loss
    )
    return model


class JohnsonSUGamlssOptunaTuner:
    """
    A class that:
      - Splits data into train/val,
      - Uses Optuna to search over hyperparams (hidden_units, L1, LR, etc.),
      - Builds a JohnsonSU model,
      - Stores the final best model.
    """
    def __init__(
        self,
        n_trials=10,
        val_size=0.2,
        random_state=42,
        max_epochs=50,
        patience=10
    ):
        """
        n_trials: how many Optuna trials
        val_size: fraction for train/val split
        random_state: for reproducibility
        max_epochs: max training epochs per trial
        patience: early stopping patience
        """
        self.n_trials = n_trials
        self.val_size = val_size
        self.random_state = random_state
        self.max_epochs = max_epochs
        self.patience = patience

        self.study_ = None
        self.best_params_ = None
        self.best_model_ = None

    def _build_and_train(self, trial):
        # Hyperparams to tune:
        hidden_units = trial.suggest_int("hidden_units", 8, 64, step=8)
        l1_strength  = trial.suggest_float("l1_strength", 1e-7, 1e-2, log=True)
        learning_rate= trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)

        # Build model
        model = build_johnson_su_model(
            n_features=self.X_train_.shape[1],
            hidden_units=hidden_units,
            l1_strength=l1_strength,
            learning_rate=learning_rate
        )

        # Early stopping
        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )
        # Fit
        model.fit(
            self.X_train_, self.Y_train_,
            validation_data=(self.X_val_, self.Y_val_),
            epochs=self.max_epochs,
            batch_size=32,
            verbose=0,
            callbacks=[es]
        )

        # Evaluate on val set
        val_loss = model.evaluate(self.X_val_, self.Y_val_, verbose=0)
        return val_loss, model

    def _objective(self, trial):
        val_loss, _ = self._build_and_train(trial)
        return val_loss

    def fit(self, X, Y):
        # Train/val split
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=self.val_size, random_state=self.random_state
        )
        self.X_train_ = X_train
        self.X_val_   = X_val
        self.Y_train_ = Y_train
        self.Y_val_   = Y_val

        # Create Optuna study
        self.study_ = optuna.create_study(direction="minimize")
        self.study_.optimize(self._objective, n_trials=self.n_trials)
        self.best_params_ = self.study_.best_params

        # Re-build final model with best hyperparams & re-train on train set
        best_trial = self.study_.best_trial
        _, final_model = self._build_and_train(best_trial)
        self.best_model_ = final_model
        return self

    def evaluate_nll(self, X, Y):
        if self.best_model_ is None:
            raise RuntimeError("Must call fit() first!")
        return self.best_model_.evaluate(X, Y, verbose=0)

    def predict_parameters(self, X):
        """
        For each row in X, returns JohnsonSU parameters [loc, scale, skew, tail].
        """
        if self.best_model_ is None:
            raise RuntimeError("Must call fit() first!")
        raw_out = self.best_model_.predict(X)
        loc       = raw_out[:, 0]
        scale     = tf.nn.softplus(raw_out[:, 1])
        skew      = raw_out[:, 2]
        tail      = tf.nn.softplus(raw_out[:, 3])
        return np.column_stack([loc, scale.numpy(), skew, tail.numpy()])