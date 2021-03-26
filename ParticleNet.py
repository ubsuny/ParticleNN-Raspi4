import tensorflow as tf

def ParticleNet(Train,TrainValues,Test):
    '''Particle Neural Network: Accepts Training Data, Training Values, and Test Data.'''

    class_names = ['Expodential','Normal']

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(4, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
              
    model.fit(Train, TrainValues, epochs=10)
    
    test_loss, test_acc = model.evaluate(Train, TrainValues, verbose=2)

    print('\nTest accuracy:', test_acc)
    
    probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
    
    predictions = probability_model.predict(Test)
    
    return predictions