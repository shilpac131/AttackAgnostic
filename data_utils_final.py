import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav

def genSpoof_list( dir_meta,is_train=False,is_eval=False):
    
    d_meta = {}
    file_list=[]

    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()
        
    if (is_train):
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    
    elif(is_eval):
        print("inside is_eval")
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list

def codec_list( dir_meta,is_emd=True):
    
    d_meta1 = {}
    d_meta2 = {}
    file_list1=[]
    file_list2 = []
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_emd):
        for line in l_meta:
             file1,codec1,file2,codec2 = line.strip().split()
             file_list1.append(file1)
             file_list2.append(file2)

             d_meta1[file1] = codec1
             d_meta2[file2] = codec2

        return d_meta1,file_list1,d_meta2,file_list2



def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			
class Dataset_ASVspoof2021_eval(Dataset):
	def __init__(self, list_IDs, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.base_dir = base_dir
            self.cut=64600 # take ~4 sec audio (64600 samples)

	def __len__(self):
            return len(self.list_IDs)


	def __getitem__(self, index):
            
            utt_id = self.list_IDs[index]
            
            X, fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000)
            X_pad = pad(X,self.cut)
            x_inp = Tensor(X_pad)
            return x_inp,utt_id  

class Dataset_ASVspoof2021_eval_w2v2_plot(Dataset):
	def __init__(self, args, list_IDs1, list_IDs2, codec_1,codec_2):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''
               
            self.list_IDs1 = list_IDs1
            self.list_IDs2 = list_IDs2
            self.codec_1 = codec_1
            self.codec_2 = codec_2
            self.args=args
            self.cut=64600 # take ~4 sec audio (64600 samples)

	def __len__(self):
            print(len(self.list_IDs1))
            return len(self.list_IDs1)


	def __getitem__(self, index):

            key1 = self.list_IDs1[index]
            key2 = self.list_IDs2[index]
            print(f"key1 {key1}")
            
            codec1 = self.codec_1[key1]
            codec2 = self.codec_2[key2]

            X1 = torch.load(f"/DATA2/s22004_local/icassp_2025/ASV2021_w2v2_xlsr53_emd/layer5/eval/{key1}.pt",
            map_location='cuda:0')
            X2 = torch.load(f"/DATA2/s22004_local/icassp_2025/ASV2021_w2v2_xlsr53_emd/layer5/eval/{key2}.pt",
            map_location='cuda:0')
            return X1,X2,codec1,codec2  


class Dataset_ASVspoof2019_train_DA(Dataset):
    def __init__(self, args, list_IDs, labels, is_set, algo):
        '''self.list_IDs  : list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''
        self.is_set = is_set
        self.list_IDs = list_IDs
        self.labels = labels
        self.algo = algo
        self.args = args
        self.cut = 64600  # Take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)  # You likely want to return len(list_IDs1)

    def __getitem__(self, index):
        key1 = self.list_IDs[index]

        # Load the audio files
        X1, fs = librosa.load(f'/home/pm_students/shilpa/datasets/LA/ASVspoof2019_LA_{self.is_set}/flac/{key1}.flac',sr=16000)
        # X1, fs = librosa.load(str(self.base_dir / f"flac/{key1}.flac"), sr=16000)
        Y1 = process_Rawboost_feature(X1, fs, self.args, self.algo)
        X1_pad = pad(Y1, self.cut)
        x1_inp = Tensor(X1_pad)

        y = self.labels[key1]  # Assuming you want the label for key1
        return x1_inp, y




#--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature