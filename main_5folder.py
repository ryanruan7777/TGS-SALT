import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
#from skimage.util.montage import montage2d as montage
from skimage.morphology import binary_opening, disk, label
import gc; gc.enable() # memory is tight
from keras.losses import binary_crossentropy
import tensorflow as tf
import scipy
from skimage import io,data,color
from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img
from tqdm import tqdm_notebook #, tnrange
#import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam,SGD
import keras.backend as K
from sklearn.model_selection import StratifiedKFold 
import cv2
#%%%
def upsample(img):# not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
def downsample(img):# not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

#%%%
#### Reference  from Heng's discussion
# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63984#382657
def get_mask_type(mask):
    border = 10
    outer = np.zeros((101-2*border, 101-2*border), np.float32)
    outer = cv2.copyMakeBorder(outer, border, border, border, border, borderType = cv2.BORDER_CONSTANT, value = 1)

    cover = (mask>0.5).sum()
    if cover < 8:
        return 0 # empty
    if cover == ((mask*outer) > 0.5).sum():
        return 1 #border
    if np.all(mask==mask[0]):
        return 2 #vertical

    percentage = cover/(101*101)
    if percentage < 0.15:
        return 3
    elif percentage < 0.25:
        return 4
    elif percentage < 0.50:
        return 5
    elif percentage < 0.75:
        return 6
    else:
        return 7

def histcoverage(coverage):
    histall = np.zeros((1,8))
    for c in coverage:
        histall[0,c] += 1
    return histall



#%%%
    



ship_dir = r'/home/lingfeng/Desktop/kaggle2/data'
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')

img_size_ori = 101
img_size_target = 101
cv_total = 5

train_df = pd.read_csv(os.path.join(ship_dir, 'train.csv'), index_col="id", usecols=[0])
depths_df = pd.read_csv(os.path.join(ship_dir, 'depths.csv'), index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

len(train_df)

train_df["images"] = [np.array(load_img("../data/train/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]

train_df["masks"] = [np.array(load_img("../data/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)

train_df["coverage_class"] = train_df.masks.map(get_mask_type)

#%%
train_all = []
evaluate_all = []
skf = StratifiedKFold(n_splits=cv_total, random_state=1234, shuffle=True)
for train_index, evaluate_index in skf.split(train_df.index.values, train_df.coverage_class):
    train_all.append(train_index)
    evaluate_all.append(evaluate_index)
    print(train_index.shape,evaluate_index.shape) # the shape is slightly different in different

def add_channel(index):
    depth_channle = []
    
    b = np.array(train_df.z[index])
    for i in range(b.size):
        depth = b[i]*np.ones((101,101,1))
        depth_channle.append(depth)
        
    return np.array(depth_channle)

def get_cv_data(cv_index):
    train_index = train_all[cv_index-1]
    evaluate_index = evaluate_all[cv_index-1]
    x_train = np.array(train_df.images[train_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    y_train = np.array(train_df.masks[train_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    x_valid = np.array(train_df.images[evaluate_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    y_valid = np.array(train_df.masks[evaluate_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    #x_train_depth = add_channel(train_index)
    print(x_train.shape)
    #print(x_train_depth.shape)
    #x_train = np.concatenate((x_train, x_train_depth), axis= -1)
    #x_valid_depth = add_channel(evaluate_index)
    #x_valid = np.concatenate((x_valid, x_valid_depth), axis= -1)
    return x_train,y_train,x_valid,y_valid




#%%%%% loss function
#%%%%
def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

def precsion( y_pred,y_true):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

def IoU_positive(y_true, y_pred, eps=1e-6):

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    
    return -K.mean( (intersection + eps) / (union + eps), axis=0)
def IoU(y_true, y_pred):
    return 0.5*IoU_positive(1-y_true, 1-y_pred)+0.5*IoU_positive(y_true, y_pred)
    
    
def Jarred(y_true, y_pred, eps=1e-6):

    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    J=K.mean( (intersection + eps) / (union + eps), axis=0)
    
    return J

def dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_p_bce(in_gt, in_pred):
    return binary_crossentropy(in_gt, in_pred)#-tf.log(-IoU(in_gt, in_pred))

def tp(y_true, y_pred):
    return K.sum(y_true * y_pred, axis=[1,2,3])

def tversky_loss(y_true, y_pred, eps=0,a=1,b=1):
    return -K.mean((tp(y_true, y_pred)+eps)/(tp(y_true, y_pred)+a*tp(1-y_true, y_pred)+b*tp(y_true, 1-y_pred)+eps),axis=0)

def focal_loss(gamma=2, alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed   

def focal_loss2(gamma = 2., alpha = 0.75):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-6, 1 - 1e-6)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1. - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), K.ones_like(y_pred) * K.constant(alpha), K.ones_like(y_pred) * K.constant(1. - alpha))
        loss = K.mean(-1. * alpha_t * (1. - p_t)**gamma * K.log(p_t))
        return loss
    return focal_loss_fixed

def IoU_binary(y_true, y_pred):
    y_pred = tf.where(tf.less(y_pred, 0.5), y_pred, tf.ones_like(y_pred))
    y_pred = tf.where(tf.equal(y_pred, 1), y_pred, tf.zeros_like(y_pred))
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])+K.sum((1-y_true) * (1-y_pred), axis=[1,2,3])
    U=K.sum(tf.ones_like(y_pred), axis=[1,2,3])
    return intersection/U


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss
from keras import losses
def combine_loss(y_true, y_pred):
    return lovasz_loss(y_true, y_pred) + 0.4*IoU_binary(y_true, y_pred)


def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
#         if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
#             metric.append(0)
#             continue
#         if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
#             metric.append(0)
#             continue
#         if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
#             metric.append(1)
#             continue
        
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)

def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred >0], tf.float64)
#%%%%plot
def plot_history(history,metric_name):
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_score.plot(history.epoch, history.history[metric_name], label="Train score")
    ax_score.plot(history.epoch, history.history["val_" + metric_name], label="Validation score")
    ax_score.legend()    




    
#%%%%
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
def train_callbacks(folder_num):
    weight_path="{}_weights.best.hdf5".format('seg_model'+str(folder_num))
    
    #checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    model_checkpoint = ModelCheckpoint(weight_path,monitor='val_my_iou_metric', 
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=0.5, patience=40, min_lr=0.0001, verbose=1)
    
    
    
    
    #early_stopping2 = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=100, verbose=1)
    model_checkpoint2 = ModelCheckpoint(weight_path,monitor='val_my_iou_metric_2', 
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr2= ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=40, min_lr=0.0001, verbose=1)
    
    
    callbacks_list = [model_checkpoint,reduce_lr]
    callbacks_list2 = [model_checkpoint2,reduce_lr2]
    return callbacks_list,callbacks_list2
#from unet_VGG11 import unet
#%%%%
from resnet_unet_scse import unet

def train(train_stage,folder_num):
    batch_size = 32
    if train_stage=='BCE_loss':
        callbacks=callbacks_list
        lr = 0.001
        epochs = 350
        metric=['binary_accuracy',my_iou_metric]
        loss_fun="binary_crossentropy"
        seg_model = unet(sigmod_output=True)
    if train_stage=='L_loss':
        callbacks=callbacks_list2
        lr = 0.0005
        epochs = 200
        metric=['binary_accuracy',my_iou_metric_2]
        loss_fun=lovasz_loss
        seg_model = unet(sigmod_output=False)
        weight_path="{}_weights.best.hdf5".format('seg_model'+str(folder_num))
        seg_model.load_weights(weight_path)
        #seg_model.save('seg_model.h5')
    
    c = optimizers.adam(lr)
    #seg_model.compile(optimizer=c, loss= "binary_crossentropy", metrics=['binary_accuracy',my_iou_metric])
    seg_model.compile(optimizer=c, loss= loss_fun, metrics=metric)
    history = seg_model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks, 
                        verbose=2)
    #plot_history(history,'my_iou_metric_2')


for cv_index in range(5):
    x_train, y_train, x_valid, y_valid =  get_cv_data(cv_index+1)
    
    #%%% Data augmentation
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    #x_train_depth = np.append(x_train_depth, [np.fliplr(x) for x in x_train_depth], axis=0)
    #x_train = np.concatenate((x_train, x_train_depth), axis=-1)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
    #x_train = np.append(x_train, [np.flipud(x) for x in x_train], axis=0)
    #y_train = np.append(y_train, [np.flipud(x) for x in y_train], axis=0)
    print(x_train.shape)
    print(y_valid.shape)
    callbacks_list,callbacks_list2=train_callbacks(cv_index)
    print('folder number:' ,cv_index)
    train('BCE_loss',cv_index)#300epoch lr:0.001
    train('L_loss',cv_index)#200 epoch lr: 0.0005

