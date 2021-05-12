'''particle_net.py creates a stand-alone ordinary neural
network model that takes as input the training and testing
data sets and the number of epochs to run the model and
outputs the predictions for the testing data set.'''
import tensorflow as tf

def particle_net(train,train_values,test,epochs):
    '''Particle Neural Network: Accepts Training Data, Training Values, and Test Data.'''
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(5, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train, train_values, epochs=epochs)
    _, test_acc = model.evaluate(train,train_values)
    print('\nTest accuracy:', test_acc)
    probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test)
    return predictions
