import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


# Function to calculate adversary noise
def grad_vec_loss(model, image, label):
    image = tf.convert_to_tensor(image, dtype=tf.float32) # convert to tensor
    label = tf.convert_to_tensor(label.reshape((1, -1)), dtype=tf.float32) # convert to tensor

    loss_function = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = loss_function(label, prediction)
    gradient = tape.gradient(loss, image)

    return gradient


# Function to generate batch of images with adversary
def fgsm_adversary_generator(model, aux_x, aux_y, batch_size, alpha=5):
    while True:
        adversaries = []
        labels = []
        benign = []
        batch_num = 0
        for batch in range(batch_size):
            n = np.random.randint(0, 4999)
            label = aux_y[n]
            original = aux_x[n].reshape((1, 32, 32, 3))

            perturbations = tf.sign(grad_vec_loss(model, original, label).numpy())
            adv = original + (perturbations * alpha)
            adv = np.clip(adv, 0, 255.0)

            if np.argmax(model(adv)) != np.argmax(label):
                benign.append(aux_x[n])
                adversaries.append(adv)
                labels.append(label)
                batch_num += 1

        adversaries = np.asarray(adversaries).reshape((batch_num,32,32,3))
        labels = np.asarray(labels)
        benign = np.asarray(benign)

        yield benign, adversaries, np.argmax(labels, axis=-1)


# Function to generate batch of images with adversary
def noise_adversary_generator(model, aux_x, aux_y, batch_size, alpha=5):
    while True:
        adversaries = []
        labels = []
        benign = []
        batch_num = 0
        for batch in range(batch_size):
            n = np.random.randint(0, 4999)
            label = aux_y[n]
            original = aux_x[n].reshape((1, 32, 32, 3))

            perturbations = grad_vec_loss(model, original, label)
            r = np.random.uniform(size=perturbations.shape)
            adv = original + (tf.sign(perturbations).numpy()* r * alpha)
            adv = np.clip(adv, 0, 255.0)

            if np.argmax(model(adv)) != np.argmax(label):
                benign.append(aux_x[n])
                adversaries.append(adv)
                labels.append(label)
                batch_num += 1

        adversaries = np.asarray(adversaries).reshape((batch_num,32,32,3))
        labels = np.asarray(labels)
        benign = np.asarray(benign)

        yield benign, adversaries, np.argmax(labels, axis=-1)

"""
## Extract a random subdataset of 'sz' records
"""
def random_subdataset(x, y, sz):
    assert x.shape[0] == y.shape[0]
    perm = np.random.permutation(x.shape[0])
    perm = perm[0:sz]

    return x[perm,:].copy(), y[perm,:].copy()


"""
## Train attack models using the 'shadow training' technique of Shokri et al.
## Inputs:
##  - x_aux, y_aux: auxiliary data
##  - target_train_size: size of training data of target model
##  - create_model_fn: function to create a model of the same type as the target model
##  - train_model_fn: function to train a model of the same type as the target model [invoke as: train_model_fn(model, x, y)]
##  - num_shadow: number of shadow models (default: 4)
##  - attack_model_fn: function to create an attack model with scikit-learn
##
##  Output:
##  - attack_models: list of attack models, one per class.
"""
def shokri_attack_models(x_aux, y_aux, target_train_size, create_model_fn, train_model_fn, num_shadow=4, attack_model_fn = lambda : LogisticRegression(solver='lbfgs')):
    assert 2*target_train_size < x_aux.shape[0]

    num_classes = y_aux.shape[1]
    class_train_list = [None] * num_classes

    def add_to_list(data):
        for label in range(0, num_classes):
            dv = data[data[:,-2] == label,:]
            col_idx = [i for i in range(0, num_classes)]
            col_idx.append(num_classes+1)
            if class_train_list[label] is None:
                class_train_list[label] = dv[:,col_idx]
            else:
                class_train_list[label] = np.vstack((class_train_list[label], dv[:,col_idx]))

    for i in range(0, num_shadow):
        ## TODO ##
        ## Insert your code here to train the ith shadow model and obtain the corresponding training data for the attack model
        ## You can use random_subdataset() to sample a subdataset from aux and add_to_list() to populate 'class_train_list'

        x_sample, y_sample = random_subdataset(x_aux, y_aux, 2*target_train_size)
        inv = np.ones((target_train_size, 1))
        outv = np.zeros((target_train_size, 1))
        x_train = x_sample[:target_train_size]
        y_train = y_sample[:target_train_size]
        x_test = x_sample[target_train_size:2*target_train_size]
        y_test = y_sample[target_train_size:2*target_train_size]

        ## TRAIN MODEL
        model = create_model_fn()
        train_model_fn(model, x_train, y_train)
        
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        
        y_train_labels = np.argmax(y_train, axis=-1).reshape(-1,1)
        y_test_labels = np.argmax(y_test,axis=-1).reshape(-1,1)
        
        dataIn = np.hstack((y_train_pred, y_train_labels))
        dataIn = np.hstack((dataIn,inv))
        # print("Data", dataIn.shape)


        dataOut = np.hstack((y_test_pred, y_test_labels))
        dataOut = np.hstack((dataOut, outv))
        add_to_list(dataIn)
        add_to_list(dataOut)

    # now train the models
    attack_models = []

    for label in range(0, num_classes):
        data = class_train_list[label]
        np.random.shuffle(data)
        y_data = data[:,-1]
        x_data = data[:,:-1]

        # train attack model
        am = attack_model_fn().fit(x_data, y_data)
        attack_models.append(am)

    return attack_models

"""
## Perform the Shokri et al. attack
## Inputs:
##  - attack_models: list of attack models, one per class.
##  - x_targets, y_targets: records to attack
##  - query_target_model: function to query the target model [invoke as: query_target_model(x)]

##  Output:
##  - in_or_out_pred: in/out prediction for each target
"""
def do_shokri_attack(attack_models, x_targets, y_targets, query_target_model):

    num_classes = y_targets.shape[1]
    assert len(attack_models) == num_classes
    y_targets_labels = np.argmax(y_targets, axis=-1)

    in_or_out_pred = np.zeros((x_targets.shape[0],))

    pv = query_target_model(x_targets)
    assert pv.shape[0] == y_targets_labels.shape[0]

    for i in range(0, pv.shape[0]):
        label = y_targets_labels[i]
        assert 0 <= label < num_classes

        am = attack_models[label]
        in_or_out_pred[i] = am.predict(pv[i,:].reshape(1,-1))

    return in_or_out_pred

