'''build_base_model.py creates the base model that is used
for the stand alone base model and the adversarial model.'''
import tensorflow as tf

def build_base_model():
    '''Builds the Basic Model'''
    feature_input_name = ['Px','Py','Pz','Mass','Tau']
    input_shape = (1,1)
    inputs = [tf.keras.Input(shape=input_shape, dtype=tf.float32, name=name)
              for name in feature_input_name]
    x_layers = tf.keras.layers.concatenate(inputs = [*inputs], axis=-1, name = 'concat')
    x_layers = tf.keras.layers.Conv1D(64, 3, padding = 'same',
                                      activation='relu', name = 'conv1')(x_layers)
    x_layers = tf.keras.layers.Conv1D(64, 1, padding = 'same',
                                      activation='relu', name = 'conv2')(x_layers)
    x_layers = tf.keras.layers.Conv1D(32, 3, padding = 'same',
                                      activation='relu', name = 'conv3')(x_layers)
    x_layers = tf.keras.layers.Conv1D(32, 1, padding = 'same',
                                      activation='relu', name = 'conv4')(x_layers)
    x_layers = tf.keras.layers.Flatten(name = 'flatten')(x_layers)
    x_layers = tf.keras.layers.Dense(64, activation='relu', name = 'relu')(x_layers)
    pred = tf.keras.layers.Dense(2, activation='softmax', name = 'output')(x_layers)
    model = tf.keras.Model(inputs=[*inputs], outputs=pred)
    return model
