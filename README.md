# 2021 KNU 전자공학설계 팀프로젝트
### _Artigence_ - _딥러닝을 활용한 BCI기반 기술 응용_
![title](https://user-images.githubusercontent.com/48755184/122780292-aefd9780-d2e9-11eb-8943-8208c790def2.png)

### Members 
|Name|github_ID|
|------|------|
|박재성|wotjd0715|
|김태헌|TaeHeonKim250|
|김지혜|jihyekim213|
|장현진|nowwhy|

# Tech
![github badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)     ![python](https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white) ![tensorflow](https://img.shields.io/badge/Tensorflow-orange?style=flat-square&logo=Tensorflow&logoColor=white)       


공개 MI dataset을 이용하여 Python 및 Tensorflow를 통해 EEGNET을 학습시켜 BMI 설계 및 제작    
Open DB : [http://deepbci.korea.ac.kr/opensource/opendb/](http://deepbci.korea.ac.kr/opensource/opendb/)


# [Progress](https://github.com/KNU-BrainAI-Capstone2021/Artigence/tree/master/Plan)✨

| Date | Summary |
| ------ | ------ |
| 3/15 | 1차 아이디어 회의 |
| 3/17 | 교수님 1차 Feedback |
| 4/1 | 2차 아이디어 회의 |
| 4/2 | 교수님 2차 Feedback |
| 5/3 | 3차 아이디어 회의 |
| 5/7 | 교수님 3차 Feedback |
| 5/12 | 1차 아이디어 PT |
| 5/14 | 4차 아이디어 회의 |
| 5/17 | 교수님 4차 Feedback |
| 5/24 | 5차 아이디어 회의 |
| 5/26 | 중간점검 + 교수님 5차 Feedback |
| 5/28 | 6차 아이디어 회의 |
| 6/2 | 7차 아이디어 회의 |
| 6/7 | 8차 아이디어 회의 |
| 6/18 | 9차 아이디어 회의 |
| 6/20 | 10차 아이디어 회의 & PPT 초안 완성 |
| 6/21 | 11차 아이디어 회의 & PPT 수정 및 보완 |
| 6/22 | 최종 아이디어 회의 & PPT 최종 완성 & 발표 준비 |
| 6/23 | 1학기 최종 발표 및 교수님 최종 Feedback 및 보완 및 개선 |

# Our Workplan
![image](https://user-images.githubusercontent.com/72614541/122770462-6db4ba00-d2e0-11eb-9230-926679be6caa.png)

# Idea meeting picture
<img src = "https://user-images.githubusercontent.com/72614541/122766549-a3f03a80-d2dc-11eb-9dbb-e2bf17c39525.png" width="400px" height="400"><img src = "https://user-images.githubusercontent.com/72614541/122766578-ace10c00-d2dc-11eb-81f0-14bbf47b1005.png" width="400px" height="400">

# DataSet we used
![image](https://user-images.githubusercontent.com/72614541/122766803-e44fb880-d2dc-11eb-87b0-8b3cd64265c0.png)
![image](https://user-images.githubusercontent.com/72614541/122766822-e87bd600-d2dc-11eb-826d-2ca4c946bdd6.png)

# How to Preprocess Dataset
BPF, ASR, CAR, ICA in Matlab

![image](https://user-images.githubusercontent.com/72614541/122767216-5f18d380-d2dd-11eb-9744-5ec11db09069.png)

# NetWork we used 
EEGNET – A compact CNN architecture for EEG-based BCIs

![image](https://user-images.githubusercontent.com/72614541/122766967-182ade00-d2dd-11eb-8a50-e0b221d4769f.png)
![image](https://user-images.githubusercontent.com/72614541/122767072-38f33380-d2dd-11eb-83e7-ce075a5b49c1.png)

CNN

![image](https://user-images.githubusercontent.com/72614541/122766924-0ba68580-d2dd-11eb-8cfd-e0117c8dc8a1.png)
![image](https://user-images.githubusercontent.com/72614541/122766936-0f3a0c80-d2dd-11eb-8e68-5edfdf063bd5.png)

# How to try to improve performance in EEGnet Models?
1. epochs.get_data()*1 → epochs.get_data()*1000  
Why? Matlab에서 preprocessing한 데이터를 불러와보니 값들이 10^-4 or 10^-5 [V] 단위로 설정되어있다.  
Solution => 우리가 원하는 Action potential은 보통 70[mV]단위이므로 *1000을 해주어서 단위를 맞춰주어 Performance가 향상 되었다.  

2. Dropout rate 0.5 → 0.8 & EarlyStop  
Why? Performance를 보면 Training Accuracy가 과도하게 높아져 Validation Accuracy가 Overfitting의 결과로 떨어졌다.  
Solution => OverFitting 방지를 위해 Dropout과 Early Stopping을 실시해보았다.  

3. AverageFoolingConv size(1,16) → AverageFoolingConv size(1,8)  
   AverageFoolingConv size(1,16) → AverageFoolingConv size(1,4)  
   
4. Sampling and Hz 조절  
Why? 우리가 EEGnet에서 주로 사용하고자 하는 주파수 대역은 8~12 Hz이므로 그것을 전처리에 Low Hz, High Hz를 8Hz,12Hz로 설정해
     주면 Performance가 향상된다.  
But => 이 방법은 딥러닝의 취지에 벗어나 사용할 수 없었다.  

5. Cross Validation Testset  
   논문에서 Performance 향상을 위해 사용한 방법으로 우리도 k-fold validation을 이용해 Performance를 올릴 수 있었다.    

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
<img src = "https://user-images.githubusercontent.com/69957743/122977750-bba8eb00-d3d0-11eb-8190-f2438c22128e.jpg" width="500px"><img src = "https://user-images.githubusercontent.com/69957743/122977806-c794ad00-d3d0-11eb-82ef-a8377b9865ff.jpg" width="500px" height="900px">
 
