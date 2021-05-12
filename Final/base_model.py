'''base_model.py creates a stand-alone base model that takes
as input the training and testing data sets and the number
of epochs to run the base model and outputs the predictions
for the testing data set.'''
from build_base_model import build_base_model

def base_model(train, train_vals, test, epochs):
    '''Creates a Base Model that can be run alone to generate its own predictions.'''
    base_models = build_base_model()
    base_models.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    base_models.fit((train[:,0],train[:,1],train[:,2],train[:,3],train[:,4]),
                   train_vals, epochs=epochs)
    predict = base_models.predict((test[:,0],test[:,1],test[:,2],test[:,3],test[:,4]))
    return predict
