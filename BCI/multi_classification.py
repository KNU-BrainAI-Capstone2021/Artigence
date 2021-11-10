import sys
import numpy as np
import mne
# EEGNet-specific imports
import tensorflow as tf
sys.path.append('C:/Users/PC/Desktop/BCI')
from Model import Multi_DS_EEGNet
from sklearn.ensemble import VotingClassifier

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
        # samples = 3sec * 200Hz sampling rate
        # 10초의 input을 overlapping하여 5개의 input으로 만든다
        #in_sig = mne.filter.create_filter(in_sig,2000,l_freq=1,h_freq=50,fir_design='firwin', verbose=True)
        signal = in_sig
        signal = np.array(signal)
        signal1 = signal[:,600:1200] 
        signal2 = signal[:,800:1400] 
        signal3 = signal[:,1000:1600]
        signal4 = signal[:,1200:1800]
        signal5 = signal[:,1400:2000]
        
        # list -> np array (input, channel, sample, kernel)
        signal1 = np.reshape(signal1,(1,28,600,1))
        signal2 = np.reshape(signal2,(1,28,600,1))
        signal3 = np.reshape(signal3,(1,28,600,1))
        signal4 = np.reshape(signal4,(1,28,600,1))
        signal5 = np.reshape(signal5,(1,28,600,1))
             
        # model training        
        model = Multi_DS_EEGNet(nb_classes=4, Chans=28, Samples=600,
                       dropoutRate=0.5, kernLength=100, F1=4, D=2, F2=8,
                       )
        # find best weights
        #latest = tf.train.latest_checkpoints('./checkpoints')
        model.load_weights('C:/Users/PC/Desktop/BCI/checkpoints/epoch_013.ckpt')
        
        # prdict 
        probs1 = model.predict(signal1)
        probs2 = model.predict(signal2)
        probs3 = model.predict(signal3)
        probs4 = model.predict(signal4)
        probs5 = model.predict(signal5)

        # find max value index
        preds1 = probs1.argmax(axis = -1)
        preds2 = probs2.argmax(axis = -1)
        preds3 = probs3.argmax(axis = -1)
        preds4 = probs4.argmax(axis = -1)
        preds5 = probs5.argmax(axis = -1)

        preds=[]
        preds.append(preds1[0])
        preds.append(preds2[0])
        preds.append(preds3[0])
        preds.append(preds4[0])
        preds.append(preds5[0])
        
        for x in range(0,5):              
                print("result=",preds[x])
               
        # hard voting
        result = 0
        for x in range(0,5):              
                tmp = preds.count(x)
                high = tmp
                if(high <= tmp):
                        result = x
                
        

        # 리스트 만들어서 voting 코드 작성
        # 최종 파이널을 f.write에 쓰기

        print("Classification result: Right = 0, Left = 1 =>" , result)
        
        # save result
        f = open(save_dir, "a")
        f.write(str(result) + "\n")
        f.close()
        

