import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import cv2
from tensorflow.keras.utils import plot_model
#from train import global_learning_rate, global_batch_size
#from prepare_data import train_size, resize_x, resize_y

save_path = 'output/'
#save_path = 'C:/Users/Anthony M. Smaldone/Desktop/class_project_4_17/'

def plot_loss_curves(qcnn_loss,qcnn_train_loss,details):
    fig = plt.figure()
    plt.plot(np.arange(len(qcnn_loss)) + 1, qcnn_loss, "ro-", label="Val Loss")
    plt.plot(np.arange(len(qcnn_train_loss)) + 1, qcnn_train_loss, "bo-", label="Train Loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, details[7], 0, 4])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Test set loss")
    plt.grid(True)
    #set_title = datatype+" Loss of "+str(round(qcnn_loss[-1],3))+" on "+str(train_size)+" ("+str(resize_x)+","+str(resize_y)+") Imgs, LR: "+str(global_learning_rate)+" DR: "+str(global_dropout_rate)+", BS: "+str(global_batch_size)
    set_title = details[6]+" Loss of "+str(round(qcnn_loss[-1],3))+" on "+str(details[0])+" ("+str(details[1])+","+str(details[2])+") Imgs, LR: "+str(details[3])+", BS: "+str(details[4])
    plt.title(set_title)
    fig.savefig(save_path+"loss.png", dpi=300)
def plot_acc_curves(qcnn_acc,qcnn_train_acc,details):
    fig = plt.figure()
    plt.plot(np.arange(len(qcnn_acc)) + 1, qcnn_acc, "ro-", label="Val Acc")
    plt.plot(np.arange(len(qcnn_train_acc)) + 1, qcnn_train_acc, "bo-", label="Train Acc")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.grid()
    plt.axis([1, details[7], 0, 1])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Test set accuracy")
    #set_title = datatype+" Accuracy of "+str(round(qcnn_acc[-1],3))+" on "+str(train_size)+" ("+str(resize_x)+","+str(resize_y)+") Imgs, LR: "+str(global_learning_rate)+" DR: "+str(global_dropout_rate)+", BS: "+str(global_batch_size)
    set_title = details[6]+" Accuracy of "+str(round(qcnn_acc[-1],3))+" on "+str(details[0])+" ("+str(details[1])+","+str(details[2])+") Imgs, LR: "+str(details[3])+", BS: "+str(details[4])
    plt.title(set_title)
    fig.savefig(save_path+"acc.png", dpi=300)
#############################

def combine_imgs(model_history,details):
    img1 = cv2.imread(save_path+'acc.png')
    img2 = cv2.imread(save_path+'loss.png')
    im_v = cv2.vconcat([img1, img2])
    cv2.imwrite(save_path+'performance.png', im_v)
    img3 = cv2.imread(save_path+'performance.png',cv2.IMREAD_UNCHANGED)
    img4 = cv2.imread(save_path+'model.png',cv2.IMREAD_UNCHANGED)
    ratio = img3.shape[0]/img4.shape[0]
    width = int(img4.shape[1]*ratio)
    height = int(img4.shape[0]*ratio)
    dim = (width, height)
    resized = cv2.resize(img4, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(save_path+'/model1.png', resized)
    img5 = cv2.imread(save_path+'/model1.png')
    output = cv2.hconcat([img3,img5])
    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open(save_path+timestr+"_history.csv",'w') as f:
        print("Datatype,"+details[6],file=f)
        print("Train Size,"+str(details[0]),file=f)
        print("Test Size,"+str(details[5]),file=f)
        print("Learning Rate,"+str(details[3]),file=f)
        print("Batch Size,"+str(details[4]),file=f)
        for k in model_history.history.keys():
            print(k,file=f)
            for i in model_history.history[k]:
                print(str(i)+",",file=f)
    cv2.imwrite(save_path+'output'+timestr+'.png',output)
    os.remove(save_path+'acc.png')
    os.remove(save_path+'loss.png')
    os.remove(save_path+'model.png')
    os.remove(save_path+'model1.png')
    os.remove(save_path+'performance.png')
    
def save_output_imgs(model,history,details):
    plot_loss_curves(history.history['val_loss'],history.history['loss'],details)
    plot_acc_curves(history.history['val_accuracy'],history.history['accuracy'],details)
    plot_model(model, to_file=save_path+'model.png', show_shapes=True,show_layer_names=True)
    combine_imgs(history,details)