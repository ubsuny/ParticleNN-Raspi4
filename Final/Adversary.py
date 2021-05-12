'''adversary.py takes as input the training and testing data sets
and the number of epochs to run the adversarial model and outputs
the predictions for the testing data set.'''
import neural_structured_learning as nsl
from build_base_model import build_base_model

def adversary(train, train_vals, test, test_vals, epochs):
    ''' Creates an Adversarial Model useing the Base Model. '''
    base_model = build_base_model()
    base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    base_model.fit((train[:,0],train[:,1],train[:,2],train[:,3],
                    train[:,4]),train_vals, epochs=epochs)
    base_adv_model = build_base_model()
    adv_model = nsl.keras.AdversarialRegularization(base_adv_model,label_keys = ['labels'])
    train_adv = {
        'Px': train[:,0],
        'Py': train[:,1],
        'Pz': train[:,2],
        'Mass': train[:,3],
        'Tau': train[:,4],
        'labels': train_vals
    }
    test_adv = {
        'Px': test[:,0],
        'Py': test[:,1],
        'Pz': test[:,2],
        'Mass': test[:,3],
        'Tau': test[:,4],
        'labels': test_vals
    }
    adv_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    adv_model.fit(train_adv, epochs=epochs)
    results = adv_model.evaluate(test_adv)
    named_results = dict(zip(adv_model.metrics_names, results))
    print('\naccuracy:', named_results['sparse_categorical_accuracy'])
    predict = adv_model.predict(test_adv)

    return predict
