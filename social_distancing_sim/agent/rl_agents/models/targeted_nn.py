from typing import Tuple

from reinforcement_learning_keras.enviroments.model_base import ModelBase
from tensorflow import keras


class TargetedNN(ModelBase):

    def _model_architecture(self) -> Tuple[Tuple[keras.layers.Layer, keras.layers.Layer], keras.layers.Layer]:
        n_units = 128 * self.unit_scale

        state_input = keras.layers.Input(name='input', shape=self.observation_shape)
        target_input = keras.layers.Input(name='input', shape=(1,))
        concat_1 = keras.layers.Concatenate()[state_input, target_input]
        fc1 = keras.layers.Dense(units=int(n_units + 1), name='fc1', activation='relu')(concat_1)
        fc2 = keras.layers.Dense(units=int(n_units / 2 + 1), name='fc2', activation='relu')(fc1)
        fc3 = keras.layers.Dense(units=int(n_units / 4 + 1), name='fc3', activation='relu')(fc2)
        action_output = keras.layers.Dense(units=self.n_actions, name='output', activation=self.output_activation)(fc3)

        return (state_input, target_input), action_output
