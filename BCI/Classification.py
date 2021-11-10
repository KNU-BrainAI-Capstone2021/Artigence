import numpy as np
# EEGNet-specific imports
import tensorflow as tf
from Model import Multi_DS_EEGNet
#show device placement
#tf.debugging.set_log_device_placement(True)

#saving result path
save_dir = "C:/Users/PC/Desktop/BCI/classification_result.txt"

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering


def Network(in_sig):
        """
        input에 preprocessing-> mne
        overlapping-> input 1~5
        """
        #list 형태 
        signal = in_sig
                
        # np array 
        signal = np.array(signal)
        signal = np.reshape(signal,(1,28,600,1))           
                
        model = Multi_DS_EEGNet(nb_classes=4, Chans=28, Samples=600,
                       dropoutRate=0.5, kernLength=100, F1=4, D=2, F2=8,
                       )
        #latest = tf.train.latest_checkpoints('./checkpoints')
        model.load_weights('C:/Users/PC/Desktop/BCI/checkpoints/epoch_013.ckpt')
        
        probs = model.predict(signal)
        preds = probs.argmax(axis = -1)
        
        print("Classification result: Right = 0, Left = 1 =>" , preds)

        f = open(save_dir, "a")
        f.write(str(preds) + "\n")
        f.close()
        

