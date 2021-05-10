import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import neural_structured_learning as nsl
from collections import defaultdict

def Adversary(train, train_vals, test, test_vals, Epochs):
    
    def build_base_model():
        '''Builds the Basic Model'''
        input_shape = (1,1)
        inputs = [tf.keras.Input(shape=input_shape, dtype=tf.float32, name=name) for name in FEATURE_INPUT_NAME]
        print(inputs)
    
        x = tf.keras.layers.concatenate(inputs = [*inputs], axis=-1, name = 'concat')
        x = tf.keras.layers.Conv1D(64, 3, padding = 'same', activation='relu', name = 'conv1')(x)
        x = tf.keras.layers.Conv1D(64, 1, padding = 'same', activation='relu', name = 'conv2')(x)
        x = tf.keras.layers.Conv1D(32, 3, padding = 'same', activation='relu', name = 'conv3')(x)
        x = tf.keras.layers.Conv1D(32, 1, padding = 'same', activation='relu', name = 'conv4')(x)
        x = tf.keras.layers.Flatten(name = 'flatten')(x)
        x = tf.keras.layers.Dense(64, activation='relu', name = 'relu')(x)
        pred = tf.keras.layers.Dense(2, activation='softmax', name = 'output')(x) 
        model = tf.keras.Model(inputs=[*inputs], outputs=pred)

        return model

    FEATURE_INPUT_NAME = ['Px','Py','Pz','Mass','Tau']
    base_model = build_base_model()
    
    base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    base_model.fit((train[:,0],train[:,1],train[:,2],train[:,3],train[:,4]),train_vals, epochs=Epochs)
    
    base_adv_model = build_base_model()
    adv_model = nsl.keras.AdversarialRegularization(base_adv_model, label_keys = ['labels'])
    
    trainAdv = {
        'Px': train[:,0],
        'Py': train[:,1],
        'Pz': train[:,2],
        'Mass': train[:,3],
        'Tau': train[:,4],
        'labels': train_vals
    }

    testAdv = {
        'Px': test[:,0],
        'Py': test[:,1],
        'Pz': test[:,2],
        'Mass': test[:,3],
        'Tau': test[:,4],
        'labels': test_vals
    }
    
    adv_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    
    adv_model.fit(trainAdv, epochs=Epochs)
    
    results = adv_model.evaluate(testAdv)
    named_results = dict(zip(adv_model.metrics_names, results))
    print('\naccuracy:', named_results['sparse_categorical_accuracy'])
    
    predict = adv_model.predict(testAdv)
    
    return predict