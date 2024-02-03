#!/usr/bin/env python
# coding: utf-8

# # AIVLE스쿨 4기 DX트랙 5차 미니프로젝트 
# ## [미션#3] 중증질환 예측 모델링

# [미션] 
#  * Target : 중증질환 (뇌경색, 뇌출혈, 복부손상, 심근경색)
#  * 데이터 분석 결과를 바탕으로 Target에 영향을 주는 Feature 전처리 (함수 정의)
#  * 머신러닝/딥러닝 모델링 후 성능 비교
#  * 최적AI 모델 선정 및 저장
#  * 새로운 출동 이력에 제시된 환자의 증상을 바탕으로 중증 질환 예측 함수 정의

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Malgun Gothic'


# In[2]:


# 응급 출동 데이터 불러오기
# 파일명 : 119_emergency_dispatch.csv, encoding='cp949'
# 중증 질환이 ['심근경색', '복부손상', '뇌경색', '뇌출혈']인 데이터만 추출
# 데이터 랜덤으로 섞기

data = pd.read_csv("./119_emergency_dispatch.csv", encoding="cp949" )
desease = data[data['중증질환'].isin(['심근경색', '복부손상', '뇌경색', '뇌출혈'])].copy()

# 데이터 랜덤으로 섞기

desease = desease.sample(frac=1).reset_index(drop=True)


# ### 1) 학습용, 평가용 데이터 준비하기

# * 데이터 전처리 함수 가져오기

# In[3]:


# 미션2에서 정의한 preprocessing 전처리 함수 정의 가져와서 실행하기

def preprocessing(desease):
    desease = desease.copy()
    
    # '발열' 컬럼 구하기 : 체온이 37도 이상이면 1, 아니면 0
    desease['발열'] = [ 1 if i >= 37 else 0 for i in desease['체온'] ]

    # '고혈압' 칼럼 구하기 : 수축기 혈압이 140 이상이면 1, 아니면 0
    desease['고혈압'] = [ 1 if i >= 140 else 0 for i in desease['수축기 혈압'] ]

    # '저혈압' 칼럼 구하기 : 수축기 혈압이 90 이하이면 1, 아니면 0
    desease['저혈압'] = [ 1 if i <= 90 else 0 for i in desease['수축기 혈압'] ]
    
    X = desease[['호흡 곤란', '간헐성 경련', '설사', '기침', '출혈', '통증', '만지면 아프다', '무감각', '마비', '현기증', '졸도',
       '말이 어눌해졌다', '시력이 흐려짐', '발열', '고혈압', '저혈압']]
    
    return X


# In[4]:


# target 중증질환 값을 Y에 저장
# desease 데이터 프레임을 preprocessing 함수를 활용하여 데이터 전처리하여 필요한 feature만 X에 저장

Y = desease['중증질환']
X = preprocessing(desease)


# In[5]:


# AI 모델링을 위한 학습/검증 데이터 나누기 : train_test_split
# 데이터 분할 비율: 학습데이터 7 : 검증데이터 3
# random_state = 2023
# 변수명 : train_x, test_x, train_y, test_y

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.3, random_state = 2023)


# ### 2) 모델링

#  * 활용 모델 : DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, DNN
#  * 성능 평가 : accuracy_score

# In[6]:


## Decision Tree
## 1) 불러오기

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

## 2) 선언하기

model1 = DecisionTreeClassifier()

## 3) 학습하기

model1.fit(train_x, train_y)

## 4) 예측하기

pred1 = model1.predict(test_x)

## 5) 평가하기

accuracy_score(test_y, pred1)


# In[7]:


## RandomForest
## 1) 불러오기

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

## 2) 선언하기

model2 = RandomForestClassifier()

## 3) 학습하기

model2.fit(train_x, train_y)

## 4) 예측하기

pred2 = model2.predict(test_x)

## 5) 평가하기

accuracy_score(test_y, pred2)


# In[8]:


test_y


# In[9]:


## XGBoost
## 1) 불러오기

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

## 2) 선언하기

model3 = XGBClassifier()


## target값 라벨링하기 {'뇌경색':0, '뇌출혈':1, '복부손상':2, '심근경색':3}

train_y_labeled = train_y.map({'뇌경색':0, '뇌출혈':1, '복부손상':2, '심근경색':3})
test_y_labeled = test_y.map({'뇌경색':0, '뇌출혈':1, '복부손상':2, '심근경색':3})

## 3) 학습하기

model3.fit(train_x, train_y_labeled)

## 4) 예측하기

pred3 = model3.predict(test_x)

## 5) 평가하기

accuracy_score(test_y_labeled, pred3)


# In[10]:


## DNN
## 1) 불러오기

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.backend import clear_session
from sklearn.metrics import accuracy_score


# 메모리 정리
clear_session()

## 2) 선언하기

model_dl = Sequential()

model_dl.add(Dense(4, activation = 'softmax'))

## target값 라벨링하기 {'뇌경색':0, '뇌출혈':1, '복부손상':2, '심근경색':3}

# -> XGB 모델에서 이미 처리o

## 3) 학습하기

model_dl.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model_dl.fit(train_x, train_y_labeled, epochs = 30, verbose = 0, validation_data = (test_x, test_y_labeled))

## 4) 예측하기

pred_dl = model_dl.predict(test_x)
pred_dl = np.argmax(pred_dl, axis=1) # 예측값 행렬의 최댓값 반환


## 5) 평가하기, np.argmax(pred_DNN, axis=1)

accuracy_score(test_y_labeled, pred_dl)


# In[11]:


model_dl.summary()


# ### 3) 최적 모델 선정 및 저장

# ## 모델 별 accuracy
# - DT : 0.9242
# - RF : 0.9248
# - XGB : 0.9274
# - Sequential : 0.8919

# In[12]:


## 모델 저장하기

#머신러닝 모델인 경우
import joblib
joblib.dump(model3, '119_model_XGC.pkl')

#딥러닝 모델인 경우
# model_DNN.save('119_model_DNN.keras')


# ### 4) 새로운 출동 이력 데이터에 대한 중증질환 예측하기

# In[13]:


# 새로운 출동 이력 데이터 : 딕셔너리 형태
new_dispatch = {
    "ID" : [500001],
    "출동일시" :['2023-04-18'],
    "이름" : ['최**'],
    "성별" : ["여성"],
    "나이" : [80],
    "체온" : [37],
    "수축기 혈압" : [145],
    "이완기 혈압" : [100],
    "호흡 곤란":[0],
    "간헐성 경련":[1],
    "설사":[0],
    "기침":[0],
    "출혈":[0],
    "통증":[1],
    "만지면 아프다":[0],
    "무감각":[0],
    "마비":[1],
    "현기증":[0],
    "졸도":[1],
    "말이 어눌해졌다":[1],
    "시력이 흐려짐":[1],

}



# In[14]:


# new_dispatch 딕셔너리를 데이터 프레임으로 변환
# 변수명 : new_data

new_data = pd.DataFrame(new_dispatch)

# new_data를 preprocessing 함수를 이용하여 데이터 전처리하기
# 변수명 : new_x

new_x = preprocessing(new_data)

new_x


# In[15]:


# 모델 불러오기

# 머신러닝 모델인 경우

model_m = model3

# 딥러닝 모델인 경우

# model_d = 


# In[16]:


# 중증질환 예측하기

# 머신러닝 모델인 경우
pred_new_m = model_m.predict(new_x)
print("예측값 : ", pred_new_m)

# 딥러닝 모델인 경우
# pred_new_d = 
# print("예측값 : ", pred_new_d)


# 중증질환 명칭으로 표시하기

sym_list = ['뇌경색', '뇌출혈', '복부손상', '심근경색']

# 머신러닝 모델인 경우
print("예측 중증질환명 : ", sym_list[pred_new_m[0]])

# 딥러닝 모델인 경우
# print("예측 중증질환명 : ",)



# ### 5) 새로운 환자(출동 이력)에 대한 중증질환 예측 함수 정의하기

#  * 1. 함수 선언하기
#  * 2. 데이터 준비하기
#  * 3. 중증 질환 예측하기
#  * 4. 중증 질환명으로 반환하기

# In[17]:


# 중증질환 예측 함수 정의하기
# 함수명 : predict_disease
# 매개변수 : new_dispatch (출동 이력 데이터, 딕셔너리 형태)
# output : 중증 질환 명칭


#########################################
# 1. 함수 선언하기                       #
#########################################

def predict_disease(new_dispatch):
    
    #########################################
    # 2. 데이터 준비하기                     #
    #########################################
    
    # 중증 질환 명칭 및 라벨링 {'뇌경색':0, '뇌출혈':1, '복부손상':2, '심근경색':3}
    # 중증 질환 리스트 정의 : 라벨링 순서대로
    sym_list = ['뇌경색', '뇌출혈', '복부손상', '심근경색']
    
    # 딕셔너리 형태의 출동 이력 데이터를 데이터 프레임으로 변환
    # 변수명 : new_data
    new_data = pd.DataFrame(new_dispatch)

    # new_data를 preprocessing 함수를 이용하여 데이터 전처리된 new_x 받아오기
    # preporcessing 함수 정의 부분이 먼저 실행되어 있어야 함
    new_x = preprocessing(new_data)

    #########################################
    # 3. 중증 질환 예측하기                  #
    #########################################
      
    # 저장된 AI모델 불러오기 
    # 모델 변수명 : model_m
    model_m = model3

    # new_x를 기반으로 중증질환 예측하기
    pred_new_m = model_m.predict(new_x)

    #########################################
    # 4. 중증 질환명으로 반환하기             #
    #########################################

    # 예측된 결과를 중증질환 명칭으로 반환하기
    return sym_list[pred_new_m[0]]
    


# In[18]:


## 확인하기
# predict_disease 함수를 이용하여, 출동 이력 데이터로 중증질환 예측하기

new_dispatch = {
    "ID" : [500001],
    "출동일시" :['2023-04-18'],
    "이름" : ['최**'],
    "성별" : ["여성"],
    "나이" : [80],
    "체온" : [37],
    "수축기 혈압" : [145],
    "이완기 혈압" : [100],
    "호흡 곤란":[0],
    "간헐성 경련":[1],
    "설사":[0],
    "기침":[0],
    "출혈":[0],
    "통증":[1],
    "만지면 아프다":[0],
    "무감각":[0],
    "마비":[1],
    "현기증":[0],
    "졸도":[1],
    "말이 어눌해졌다":[1],
    "시력이 흐려짐":[1],
}


predict_disease(new_dispatch)


# ## 추가실습

# In[19]:


my_dispatch = {
    "ID" : [500001],
    "출동일시" :['2089-04-18'],
    "이름" : ['최민성'],
    "성별" : ["남성"],
    "나이" : [59],
    "체온" : [37],
    "수축기 혈압" : [138],
    "이완기 혈압" : [102],
    "호흡 곤란":[1],
    "간헐성 경련":[0],
    "설사":[0],
    "기침":[1],
    "출혈":[0],
    "통증":[1],
    "만지면 아프다":[0],
    "무감각":[0],
    "마비":[0],
    "현기증":[1],
    "졸도":[0],
    "말이 어눌해졌다":[0],
    "시력이 흐려짐":[1],
}


predict_disease(my_dispatch)


# ## 미션#3 Clear
# ## 수고하셨습니다!!
