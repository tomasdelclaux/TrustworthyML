#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project Option 1 -- part1.py

# This file contains the part1 code
"""

import sys
import os

import time

import numpy as np

import sklearn
from sklearn.metrics import confusion_matrix
from efficientnet.tfkeras import EfficientNetB0

# we'll use tensorflow and keras for neural networks
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras

import utils # we need this
import attacks
import nets
import pandas as pd
import openpyxl

import cv2# for the denoising defenses
    
"""
## Plots an adversarial perturbation, i.e., original input orig_x, adversarial example adv_x, and the difference (perturbation)
"""
def plot_adversarial_example(pred_fn, orig_x, adv_x, labels, fname='adv_exp.png', show=True, save=True):
    perturb = adv_x - orig_x
    
    # compute confidence
    in_label, in_conf = utils.pred_label_and_conf(pred_fn, orig_x)
    
    # compute confidence
    adv_label, adv_conf = utils.pred_label_and_conf(pred_fn, adv_x)
    
    titles = ['{} (conf: {:.2f})'.format(labels[in_label], in_conf), 'Perturbation',
              '{} (conf: {:.2f})'.format(labels[adv_label], adv_conf)]
    
    images = np.r_[orig_x, perturb, adv_x]
    
    # plot images
    utils.plot_images(images, fig_size=(8,3), titles=titles, titles_fontsize=12,  out_fp=fname, save=save, show=show)  


######### Prediction Fns #########

"""
## Basic prediction function
"""
def basic_predict(model, x):
    return model.predict(x)


#### TODO: implement your defense(s) as a new prediction function
#### Put your code here
"""
## Gaussian Blur prediction function
"""
def Gaussian_blur_filter(model, x):

    ## NEED TO INCLUDE PARAMETERS AS ARGUMETS TO THE FUNCTION (KSIZE AND SIGMAX, SIGMAY)
    dst = np.array([cv2.GaussianBlur(np.uint8(image*255), (5, 5), 1, 1)/255 for image in x])

    ## UNCOMMENT FOR IMAGE COMPARISON
    # cv2.imshow("BEFORE", x[0])
    # cv2.imshow("AFTER", dst[0])

    # cv2.imwrite('./figures/gaussian_blur_before.jpg', x[0]*255)
    # cv2.imwrite('./figures/gaussian_blur_after.jpg', dst[0]*255)
    # cv2.waitKey(0)

    return model(dst)

"""
## Median Blur prediction function
"""
def Median_blur_filter(model, x):

    ## NEED TO ADD PARAMETERS KSIZE
    dst = np.array([cv2.medianBlur(np.uint8(image*255), 3)/255 for image in x])

    # # UNCOMMENT FOR IMAGE COMPARISON
    # cv2.imshow("BEFORE", x[0])
    # cv2.imshow("AFTER", dst[0])

    # cv2.imwrite('./figures/median_blur_before.jpg', x[0]*255)
    # cv2.imwrite('./figures/median_blur_after.jpg', dst[0]*255)
    # cv2.waitKey(0)
    
    return model(dst)

"""
## Laplace Noise prediction function
"""
def Laplace_noise(model,x, sigma=20):
    noise = np.random.laplace(0.0,sigma,size=tf.shape(x))
    x_noisy = x*255 + noise

    x_noisy_clipped = tf.clip_by_value(x_noisy, 0, 255.0)/255

    # # UNCOMMENT FOR IMAGE COMPARISON
    # cv2.imshow("BEFORE", x[0])
    # cv2.imshow("AFTER", x_noisy_clipped[0].numpy())

    # cv2.imwrite('./figures/laplace_before.jpg', x[0]*255)
    # cv2.imwrite('./figures/laplace_after.jpg', x_noisy_clipped[0].numpy()*255)
    # cv2.waitKey(0)
    return model(x_noisy_clipped)


"""
## Gaussian Noise prediction function
"""
def Gaussian_noise(model, x, sigma=20):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)
    x_noisy = x*255 + noise

    #clip
    x_noisy_clipped = tf.clip_by_value(x_noisy, 0, 255.0)/255

    # # UNCOMMENT FOR IMAGE COMPARISON
    # cv2.imshow("BEFORE", x[0])
    # cv2.imshow("AFTER", x_noisy_clipped[0].numpy())

    # cv2.imwrite('./figures/gauss_before.jpg', x[0]*255)
    # cv2.imwrite('./figures/gauss_after.jpg', x_noisy_clipped[0].numpy()*255)
    # cv2.waitKey(0)
    return model(x_noisy_clipped)


"""
## Non-local Means Denoising algorithm for smoothing
"""
def deNoise_filter(model, x, f,c,k):

    ## NEED TO ADD PARAMETERS TO THE FUNCTIONA ARGUMENTS FOR DENOISING
    dst = np.array([cv2.fastNlMeansDenoisingColored(np.uint8(image*255),None,f,c,k,21)/255 for image in x])

    # # UNCOMMENT FOR IMAGE COMPARISON
    # cv2.imshow("BEFORE", x[0])
    # cv2.imshow("AFTER", dst[0])

    # cv2.imwrite('./figures/nonLocal_before.jpg', x[0]*255)
    # cv2.imwrite('./figures/nonLocal_after.jpg', dst[0]*255)
    # cv2.waitKey(6000)
    return model.predict(dst)

"""
## Non-local Means Denoising algorithm for smoothing
"""
def combined_defense(model,x,sigma=8):
    noise = np.random.laplace(0.0,sigma,size=tf.shape(x))

    deNoise = np.array([cv2.fastNlMeansDenoisingColored(np.uint8(image*255),None,8,1,7,21)/255 for image in x])
    x_noisy = deNoise*255 + noise
    x_noisy_clipped = tf.clip_by_value(x_noisy, 0, 255.0)/255
    # # UNCOMMENT FOR IMAGE COMPARISON
    # cv2.imshow("BEFORE", x[0])
    # cv2.imshow("AFTER", dst[0])
    # cv2.waitKey(0)
    return model.predict(x_noisy_clipped)


######### Membership Inference Attacks (MIAs) #########

"""
## A very simple threshold-based MIA
"""
def simple_conf_threshold_mia(predict_fn, x, y=None, thresh=0.9999, val_x=None, val_y=None, num_classes=10):   
    pred_y = predict_fn(x)
    pred_y_conf = np.max(pred_y, axis=-1)
    return (pred_y_conf > thresh).astype(int)

    
def shokri_attack(predict_fn, x, y=None, thresh=None, val_x=None, val_y=None, num_classes=10):
    from sklearn.linear_model import LogisticRegression
    attack_model_fn = lambda : LogisticRegression(solver='lbfgs')
    target_model_train_fn = lambda: nets.get_simple_classifier(num_hidden=128, num_labels=num_classes)
    create_model_fn = target_model_train_fn
    train_model_fn = lambda model, x, y: nets.train_model(model, x, y, None, None, 500, verbose=False)
    val_x = val_x.reshape(val_x.shape[0], 32*32*3)
    attack_models = attacks.shokri_attack_models(val_x, val_y, 2499, create_model_fn, train_model_fn, num_shadow=num_classes, attack_model_fn=attack_model_fn)

    num_classes = y.shape[1]
    assert len(attack_models) == num_classes
    y_targets_labels = np.argmax(y, axis=-1)
    in_or_out_pred = np.zeros((x.shape[0],))

    pv = predict_fn(x)

    assert pv.shape[0] == y_targets_labels.shape[0]

    for i in range(0, pv.shape[0]):
        label = y_targets_labels[i]
        assert 0 <= label < num_classes

        am = attack_models[label]
        in_or_out_pred[i] = am.predict(pv[i,:].reshape(1,-1))

    return in_or_out_pred

######### Adversarial Examples #########

  
#### TODO [optional] implement new adversarial examples attacks.
#### Put your code here  
#### Note: you can have your code save the data to file so it can be loaded and evaluated in Main() (see below).
def create_adversarial_fgsm_examples(model, test_x, test_y, data_name, model_name, model_type, batch_size=100):
    samples_fp = 'advexp_{}_{}_{}_1.npz'.format(data_name,model_name,model_type)

    if not os.path.isfile(samples_fp):
        x_benign, x_adv_samples, correct_labels = next(attacks.fgsm_adversary_generator(model, test_x, test_y, batch_size, alpha=0))
        np.savez_compressed(samples_fp, benign_x=x_benign, benign_y=correct_labels, adv_x=x_adv_samples)
    
    samples_fp = 'advexp_{}_{}_{}_2.npz'.format(data_name,model_name,model_type)

    if not os.path.isfile(samples_fp):
        x_benign, x_adv_samples, correct_labels = next(attacks.fgsm_adversary_generator(model, test_x, test_y, batch_size, alpha=0.01))
        np.savez_compressed(samples_fp, benign_x=x_benign, benign_y=correct_labels, adv_x=x_adv_samples)
    
    samples_fp = 'advexp_{}_{}_{}_3.npz'.format(data_name,model_name,model_type)

    if not os.path.isfile(samples_fp):
        x_benign, x_adv_samples, correct_labels = next(attacks.fgsm_adversary_generator(model, test_x, test_y, batch_size, alpha=0.02))
        np.savez_compressed(samples_fp, benign_x=x_benign, benign_y=correct_labels, adv_x=x_adv_samples)

    samples_fp = 'advexp_{}_{}_{}_4.npz'.format(data_name,model_name,model_type)

    if not os.path.isfile(samples_fp):
        x_benign, x_adv_samples, correct_labels = next(attacks.noise_adversary_generator(model, test_x, test_y, batch_size, alpha=0.01))
        np.savez_compressed(samples_fp, benign_x=x_benign, benign_y=correct_labels, adv_x=x_adv_samples)

    return

   
######### Main() #########
   
if __name__ == "__main__":

    # Let's check our software versions
    print('### Python version: ' + __import__('sys').version)
    print('### NumPy version: ' + np.__version__)
    print('### Scikit-learn version: ' + sklearn.__version__)
    print('### Tensorflow version: ' + tf.__version__)
    print('### TF Keras version: ' + keras.__version__)
    print('------------')

    
    data_name = sys.argv[1]
    model_name = sys.argv[2]
    model_type = sys.argv[3]

    if data_name == "cifar10":
        num_classes = 10
    
    elif data_name == "cifar100":
        if model_name == "coarse":
            data_file = "./Data/cifar100_coarse_data.npz"
            model_file = "./Models/cifar100_coarse_{}.h5".format(model_type)
            num_classes = 20
        else:
            data_file = "./Data/cifar100_fine_data.npz"
            model_file = "./Models/cifae100_fine_{}.h5".format(model_type)
            num_classes = 100

    f_filter = int(sys.argv[4])
    c_filter = int(sys.argv[5])
    k_size = int(sys.argv[6])

    # global parameters to control behavior of the pre-processing, ML, analysis, etc.
    seed = 42

    # deterministic seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # keep track of time
    st = time.time()

    #### load the data
    print('\n------------ Loading Data & Model ----------')
    
    train_x, train_y, test_x, test_y, val_x, val_y, labels = utils.load_data(fp=data_file,num_labels=num_classes)
    num_class= len(labels)
    assert num_classes == num_classes # cifar10
    
    ### load the target model (the one we want to protect)
    target_model_fp = model_file

    model, _ = utils.load_model(target_model_fp)
    ## model.summary() ## you can uncomment this to check the model architecture (ResNet)
    
    st_after_model = time.time()
        
    ### let's evaluate the raw model on the train and test data
    train_loss, train_acc_raw = model.evaluate(train_x, train_y, verbose=0)
    test_loss, test_acc_raw = model.evaluate(test_x, test_y, verbose=0)
    print('[Raw Model] Train accuracy: {:.2f}% --- Test accuracy: {:.2f}%'.format(100*train_acc_raw, 100*test_acc_raw))
    
    ### let's wrap the model prediction function so it could be replaced to implement a defense
    # predict_fn = lambda x: basic_predict(model, x)
    # predict_fn = lambda x: Gaussian_blur_filter(model, x)
    # predict_fn = lambda x: Median_blur_filter(model, x)
    predict_fn = lambda x: deNoise_filter(model, x, f_filter, c_filter, k_size)
    # predict_fn = lambda x: Gaussian_noise(model, x)
    # predict_fn = lambda x: Laplace_noise(model, x)
    # predict_fn = lambda x: combined_defense(model, x)
    
    ### now let's evaluate the model with this prediction function
    pred_y = predict_fn(train_x)
    train_acc = np.mean(np.argmax(train_y, axis=-1) == np.argmax(pred_y, axis=-1))
    
    pred_y = predict_fn(test_x)
    test_acc = np.mean(np.argmax(test_y, axis=-1) == np.argmax(pred_y, axis=-1))
    print('[Model] Train accuracy: {:.2f}% --- Test accuracy: {:.2f}%'.format(100*train_acc, 100*test_acc))
        
    
    ### evaluating the privacy of the model wrt membership inference
    mia_eval_size = 2000
    mia_eval_data_x = np.r_[train_x[0:mia_eval_size], test_x[0:mia_eval_size]]
    mia_eval_data_y = np.r_[train_y[0:mia_eval_size], test_y[0:mia_eval_size]]
    mia_eval_data_in_out = np.r_[np.ones((mia_eval_size,1)), np.zeros((mia_eval_size,1))]
    assert mia_eval_data_x.shape[0] == mia_eval_data_in_out.shape[0]
    
    # so we can add new attack functions as needed
    print('\n------------ Privacy Attacks ----------')
    mia_attack_fns = []
    mia_attack_fns.append(('Simple MIA Attack', simple_conf_threshold_mia))
    mia_attack_fns.append(('Shokri MIA Attack', shokri_attack))
    
    dfPriv = pd.DataFrame()
    for i, tup in enumerate(mia_attack_fns):
        attack_str, attack_fn = tup
        
        in_out_preds = attack_fn(predict_fn, mia_eval_data_x, mia_eval_data_y, val_x=val_x, val_y=val_y, num_classes=num_classes).reshape(-1,1)
        assert in_out_preds.shape == mia_eval_data_in_out.shape, 'Invalid attack output format'
        
        cm = confusion_matrix(mia_eval_data_in_out, in_out_preds, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        
        attack_acc = np.trace(cm) / np.sum(np.sum(cm))
        attack_adv = tp / (tp + fn) - fp / (fp + tn)
        attack_precision = tp / (tp + fp)
        attack_recall = tp / (tp + fn)
        attack_f1 = tp / (tp + 0.5*(fp + fn))
        print('{} --- Attack accuracy: {:.2f}%; advantage: {:.3f}; precision: {:.3f}; recall: {:.3f}; f1: {:.3f}'.format(attack_str, attack_acc*100, attack_adv, attack_precision, attack_recall, attack_f1))
        dfPriv=dfPriv.append({"Accuracy": attack_acc*100, "Advantage": attack_adv, "Precision": attack_precision, "Recall": attack_recall, "F1": attack_f1}, ignore_index=True)


    ### evaluating the robustness of the model wrt adversarial examples
    print('\n------------ Adversarial Examples ----------')
    create_adversarial_fgsm_examples(model, test_x, test_y, data_name, model_name, model_type)
    advexp_fps = []
    advexp_fps.append(('Adversarial examples attack0', 'advexp_{}_{}_{}_1.npz'.format(data_name,model_name,model_type)))
    advexp_fps.append(('Adversarial examples attack1', 'advexp_{}_{}_{}_2.npz'.format(data_name,model_name,model_type)))
    advexp_fps.append(('Adversarial examples attack2', 'advexp_{}_{}_{}_3.npz'.format(data_name,model_name,model_type)))
    advexp_fps.append(('Adversarial examples attack3', 'advexp_{}_{}_{}_4.npz'.format(data_name,model_name,model_type)))

    dfAdv = pd.DataFrame()
    for i, tup in enumerate(advexp_fps):
        attack_str, attack_fp = tup
        
        data = np.load(attack_fp)
        adv_x = data['adv_x']
        benign_x = data['benign_x']
        benign_y = data['benign_y']

        ## Uncomment to see benign and adversary images
        # fig, (ax1,ax2) = plt.subplots(1, 2, sharey=True)
        # ax1.imshow(benign_x[0].reshape(32,32, 3))
        # ax1.set_title("Original Image")
        # ax2.imshow(adv_x[0].reshape(32,32, 3))
        # ax2.set_title("Image with Adversary")
        plt.show()
        
        benign_pred_y = predict_fn(benign_x)
        benign_acc = np.mean(benign_y == np.argmax(benign_pred_y, axis=-1))
        
        adv_pred_y = predict_fn(adv_x)
        adv_acc = np.mean(benign_y == np.argmax(adv_pred_y, axis=-1))

        dfAdv=dfAdv.append({"Benign acc": benign_acc*100, "Advantage": 100*adv_acc}, ignore_index=True)
        print('{} --- Benign accuracy: {:.2f}%; adversarial accuracy: {:.2f}%'.format(attack_str, 100*benign_acc, 100*adv_acc))
        
    print('------------\n')

    et = time.time()
    
    print('Elapsed time -- total: {:.1f} seconds (data & model loading: {:.1f} seconds)'.format(et - st, st_after_model - st))

    dfRaw = pd.DataFrame([[train_acc_raw*100, test_acc_raw*100]], index=['raw'], columns=["Train Acc", "Test Acc"])
    dfModel = pd.DataFrame([[train_acc*100, test_acc*100]], index=['model'], columns=["Train Acc", "Test Acc"])
    # df = pd.DataFrame()
    print(dfRaw)
    print(dfPriv)
    path = './Results/{}_{}_{}_data.xlsx'.format(data_name, model_name, model_type)
    with pd.ExcelWriter(path) as writer:
        dfRaw.to_excel(writer, sheet_name='raw_{}'.format(model_name))
        dfModel.to_excel(writer, sheet_name='model_{}'.format(model_name))
        dfPriv.to_excel(writer, sheet_name='privacy_{}'.format(model_name))
        dfAdv.to_excel(writer, sheet_name='adversary_{}'.format(model_name))
    sys.exit(0)
