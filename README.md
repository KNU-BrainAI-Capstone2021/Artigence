# 2021 KNU 전자공학설계 팀프로젝트
### _Artigence_ - _딥러닝을 활용한 BCI기반 기술 응용_
![title](https://user-images.githubusercontent.com/48755184/122780292-aefd9780-d2e9-11eb-8943-8208c790def2.png)

### Members 
|Name|github_ID|
|------|------|
|박재성|wotjd0715|
|김태헌|TaeHeonKim250|
|장현진|nowwhy|
|신예주| |
|김민채| |
|김예지| |
|김지혜|jihyekim213|

# Code
### [Deep_learning_model](https://github.com/KNU-BrainAI-Capstone2021/Artigence/tree/master/Deep_learning_model) 참고   
**Cross_Newmodel.py** : Cross-subject performance with proposed Multi-DS-EEGNet    
**Cross_Subject.py**  : Cross-subject performance with EEGNet   
**NewModel.py**       : Within-subject performance with proposed Multi-DS-EEGNet    
**Within_Subject.py** : Within-subject performance with EEGNet   
**EEGNet.py**         : Several EEGNet version with proposed Multi-DS-EEGNet    
**preprocessing.m**   : Matlab code for preprocessing(BPF,ASR,CAR,ICA) with eeglab   

# Tech
![github badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)     ![python](https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white) ![tensorflow](https://img.shields.io/badge/Tensorflow-orange?style=flat-square&logo=Tensorflow&logoColor=white)       


공개 MI dataset을 이용하여 Python 및 Tensorflow를 통해 EEGNET을 학습시켜 BMI 설계 및 제작    
Open DB : [http://deepbci.korea.ac.kr/opensource/opendb/](http://deepbci.korea.ac.kr/opensource/opendb/)


# Progress ✨
### [_Summary_](https://github.com/KNU-BrainAI-Capstone2021/Artigence/tree/master/Summary) 참고

# Our Workplan
## 2학기
![image](https://user-images.githubusercontent.com/69957743/145677615-e26d222e-b702-44da-a953-78c92ab290cd.png)

## 1학기
![image](https://user-images.githubusercontent.com/69957743/145677648-6f3212d4-7683-418b-b5a4-b1e994ab2843.png)

## Project 진행상황
![image](https://user-images.githubusercontent.com/69957743/145677158-d44e549e-6777-4fdb-875b-c6bd04a603c0.png)
![image](https://user-images.githubusercontent.com/69957743/145677118-07015831-3bdd-4d0a-a499-07f82a45eda2.png)

# Idea meeting picture
<img src = "https://user-images.githubusercontent.com/72614541/122766549-a3f03a80-d2dc-11eb-9dbb-e2bf17c39525.png" width="400px" height="400"><img src = "https://user-images.githubusercontent.com/72614541/122766578-ace10c00-d2dc-11eb-81f0-14bbf47b1005.png" width="400px" height="400">

# DataSet we used
![image](https://user-images.githubusercontent.com/72614541/122766803-e44fb880-d2dc-11eb-87b0-8b3cd64265c0.png)
![image](https://user-images.githubusercontent.com/72614541/122766822-e87bd600-d2dc-11eb-826d-2ca4c946bdd6.png)

# BCI System process
![image](https://user-images.githubusercontent.com/69957743/145677273-7fc0c40d-781a-4ef6-9473-925d26b4727b.png)

# How to Preprocess Dataset
BPF, ASR, CAR, ICA in Matlab

![image](https://user-images.githubusercontent.com/72614541/122767216-5f18d380-d2dd-11eb-9744-5ec11db09069.png)

# Preprocessing  
## BPF(Band Pass Filter)  
50Hz이상 주파수 제거  
![image](https://user-images.githubusercontent.com/69957743/122971652-3b7f8700-d3ca-11eb-903a-4fdf37cc291f.png)![image](https://user-images.githubusercontent.com/69957743/122971704-4a663980-d3ca-11eb-96d3-fdd99f90cbb3.png)
  
## ASR(Artifact Subspace Reconstruction)  
bad channel 제거  
<img src = "https://user-images.githubusercontent.com/69957743/122971811-6ec21600-d3ca-11eb-8e21-c0bc1f467a99.jpg" width="500px"><img src = "https://user-images.githubusercontent.com/69957743/122971858-7aadd800-d3ca-11eb-96da-c00aa35f4e11.jpg" width="500px">

## CAR(Common Average Reference)  
전위 기준점 변경  
<img src = "https://user-images.githubusercontent.com/69957743/122971919-8a2d2100-d3ca-11eb-9dea-1ab91ebd197b.jpg" width="500px"><img src = "https://user-images.githubusercontent.com/69957743/122971957-95804c80-d3ca-11eb-9f5c-ea299ddbf11c.jpg" width="500px">  
    
## ICA(Independent Component Analysis)  
Noise가 될 수 있는 요인 제거 ex) Eye Blinking, Heart Rate...  
<img src = "https://user-images.githubusercontent.com/69957743/122977750-bba8eb00-d3d0-11eb-8190-f2438c22128e.jpg" width="500px"><img src = "https://user-images.githubusercontent.com/69957743/122977806-c794ad00-d3d0-11eb-82ef-a8377b9865ff.jpg" width="500px" height="500px">
 


