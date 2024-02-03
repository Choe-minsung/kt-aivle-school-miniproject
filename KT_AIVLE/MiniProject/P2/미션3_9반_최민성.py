#!/usr/bin/env python
# coding: utf-8

# # **Mission 3_주차 등록 수요 데이터 분석**

# ## <미션>

# * 1. 데이터 기초 정보 확인하기
#     - 기초 통계량, NaN 값 확인 등 기본 분석 수행
# * 2. 단변량 분석 
#     - 수치형, 범주형 데이터
#     - 단일 변수로 분석 : 등록차량수, 총세대수, 버스정류장수, 지하철역수, 공가수, 임대료, 보증금
#     - 여러 변수를 묶어서 분석 : 건물구분, 공급유형
# * 3. 이변량 분석
#     - 수치형 vs 수치형
#         - 전체 상관계수를 구하고 시각화
#         - 상관계수가 높은 변수에 대한 산점도를 구하기
#     - 범주형 vs 수치형
#         - 범주 그룹간에 평균의 차이가 있는지 검증
#         - bar 그래프를 통해 평균의 차이를 시각화

# ## <환경설정 >

# ### &nbsp;&nbsp; 1) 라이브러리 불러오기

# * 기본적으로 필요한 라이브러리를 import 하도록 코드가 작성되어 있습니다.
# * 필요하다고 판단되는 라이브러리를 추가하세요.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 아래 필요한 라이브러리, 함수를 추가하시오.
## 코드 입력
import scipy.stats as spst


# ### &nbsp;&nbsp; 2) 한글 폰트 설정하기

# In[2]:


plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False


# ### &nbsp;&nbsp; 3) 데이터 불러오기

# * 파일명 : registerd_parking_preprocessed.csv
# * data 변수에 저장하기 

# In[3]:


## 코드 입력
data = pd.read_csv('registerd_parking_preprocessed.csv')


# ## 1. 데이터 기초 정보 확인하기

# - 데이터의 양, 컬럼명, 데이터 타입 확인하기
# - 데이터프레임 전체에 대한 기초통계량 구하기
# - 결측치가 있는지 확인하고, 결측치가 있는 경우 조치하기
# 
# 

# #### &nbsp;&nbsp; 1-1) 데이터프레임 크기 확인

# In[4]:


## shape
## 코드 입력
data.shape



# #### &nbsp;&nbsp; 1-2) 컬럼명, 데이터 개수, 데이터 타입 확인

# In[5]:


## info
## 코드 입력
data.info()


# #### &nbsp;&nbsp; 1-3) 기초통계 확인

# In[6]:


## describe
## 코드 입력
data.describe()


# #### &nbsp;&nbsp; 1-4) 결측치(N/A) 개수 확인

# In[7]:


## 결측치가 있는지 확인하기 : isna
## 코드 입력
data.isna().sum()


# In[8]:


## 결측치가 있는 컬럼은 무엇인가요?

# 지하철역수       18
# 버스정류장수      1


# In[9]:


data.head(2)


# In[10]:


## 결측치 처리 : dropna, fillna
## 코드 입력

# dropna
# data.dropna(subset = ['지하철역수', '버스정류장수'], axis = 0)

# fillna(0)
data.fillna(0)


# In[12]:


## 결측치 처리 결과 확인하기 : info
## 코드 입력
data.info()


# <br><br><hr>

# ##  2. 단변량 분석
# 
# 

# #### &nbsp;&nbsp; 2-1)  등록 차량수

# In[13]:


## '등록차량수' 변수의 기초통계량 확인
## 코드 입력
display(data[['등록차량수']].describe().T)

## 2행 1열 그래프 그리기
## '등록차량수' 변수의 histplot그리기
## 코드 입력
plt.figure(figsize = (8,6))

plt.subplot(2,1,1)
sns.histplot(x = data['등록차량수'])
plt.grid()

## '등록차량수' 변수의 boxplot 그리기
## 코드 입력
plt.subplot(2,1,2)
sns.boxplot(x = data['등록차량수'])
plt.grid()

plt.show()


# In[55]:


## boxplot의 whiskers 값 구하기
## 코드 입력
lower_bound = data.loc[data['등록차량수'] >= data['등록차량수'].describe()['25%'] - (data['등록차량수'].describe()['75%'] - data['등록차량수'].describe()['25%']) * 1.5, '등록차량수'].min()
upper_bound = data.loc[data['등록차량수'] <= data['등록차량수'].describe()['75%'] + (data['등록차량수'].describe()['75%'] - data['등록차량수'].describe()['25%']) * 1.5, '등록차량수'].max()

lower_bound, upper_bound


# #### &nbsp;&nbsp; 2-2) 수치형 데이터

# * 기초 통계량 분석
# * 그래프 : boxplot, histplot

# ##### &nbsp; &nbsp; &nbsp; &nbsp; 2-2-1) 총 세대수

# In[15]:


## 기초통계량 확인
## 코드 입력
display(data[['총세대수']].describe().T)

##  histplot
## 코드 입력
plt.figure(figsize = (8,6))
plt.subplot(2,1,1)
sns.histplot(x = data['총세대수'])
plt.grid()

## boxplot 
## 코드 입력
plt.subplot(2,1,2)
sns.boxplot(x = data['총세대수'])
plt.grid()

plt.show()



# ##### &nbsp; &nbsp; &nbsp; &nbsp; 2-2-2) 공가수

# In[16]:


## 기초통계량 확인
## 코드 입력
display(data[['공가수']].describe().T)

##  histplot
## 코드 입력
plt.figure(figsize = (8,6))
plt.subplot(2,1,1)
sns.histplot(x = data['공가수'])
plt.grid()

## boxplot 
## 코드 입력
plt.subplot(2,1,2)
sns.boxplot(x = data['공가수'])
plt.grid()

plt.show()



# ##### &nbsp; &nbsp; &nbsp; &nbsp; 2-2-3) 지하철역 수

# In[17]:


## 기초통계량 확인
## 코드 입력
display(data[['지하철역수']].describe().T)

##  histplot
## 코드 입력
plt.figure(figsize = (8,6))
plt.subplot(2,1,1)
sns.histplot(x = data['지하철역수'])
plt.grid()

## boxplot 
## 코드 입력
plt.subplot(2,1,2)
sns.boxplot(x = data['지하철역수'])
plt.grid()

plt.show()



# ##### &nbsp; &nbsp; &nbsp; &nbsp; 2-2-4) 버스정류장수

# In[18]:


## 기초통계량 확인
## 코드 입력
display(data[['버스정류장수']].describe().T)

##  histplot
## 코드 입력
plt.figure(figsize = (8,6))
plt.subplot(2,1,1)
sns.histplot(x = data['버스정류장수'])
plt.grid()

## boxplot 
## 코드 입력
plt.subplot(2,1,2)
sns.boxplot(x = data['버스정류장수'])
plt.grid()

plt.show()



# ##### &nbsp; &nbsp; &nbsp; &nbsp; 2-2-5) 임대료

# In[19]:


## 기초통계량 확인
## 코드 입력
display(data[['임대료']].describe().T)

##  histplot
## 코드 입력
plt.figure(figsize = (8,6))
plt.subplot(2,1,1)
sns.histplot(x = data['임대료'])
plt.grid()

## boxplot 
## 코드 입력
plt.subplot(2,1,2)
sns.boxplot(x = data['임대료'])
plt.grid()

plt.show()



# ##### &nbsp; &nbsp; &nbsp; &nbsp; 2-2-6) 임대보증금

# In[20]:


## 기초통계량 확인
## 코드 입력
display(data[['임대보증금']].describe().T)

##  histplot
## 코드 입력
plt.figure(figsize = (8,6))
plt.subplot(2,1,1)
sns.histplot(x = data['임대보증금'])
plt.grid()

## boxplot 
## 코드 입력
plt.subplot(2,1,2)
sns.boxplot(x = data['임대보증금'])
plt.grid()

plt.show()



# #### &nbsp;&nbsp; 2-3) 수치형 데이터 - pivot 테이블로 구성된 데이터를 재구성하여 분석하기

# * pd.melt 활용
#     * pivot table 형태로 구성된 데이터프레임을 구분자(variable)와 값(value)로 재구성하기
#     * pd.melt(dataframe명, id_vars = [identifier로 사용될 컬럼명], value_vars = [구분자로 쓰일 컬럼명] )
# * 기초 통계량 분석
# * 그래프 : boxplot, barplot

# ##### &nbsp; &nbsp; &nbsp; &nbsp; 2-3-1) 전용면적 구간별 세대수

# In[21]:


data.columns


# In[ ]:





# In[26]:


## 단지코드와 전용면적에 관련된 column 추출하기 
## area_data 변수에 저장
## 코드 입력

area_data = data[['단지코드', '전용면적_10_30', '전용면적_30_40', 
                     '전용면적_40_50', '전용면적_50_100', '전용면적_100이상']]
area_data


# In[30]:


## 전용면적에 관련된 컬럼들을 pd.melt로 재구성하기
## melt_area_data 변수에 저장

## 코드 입력

# pivot table 형태로 구성된 데이터프레임을 구분자(variable)와 값(value)로 재구성하기
# pd.melt(dataframe명, id_vars = [identifier로 사용될 컬럼명],
#         value_vars = [구분자로 쓰일 컬럼명] )

melt_area_data = pd.melt(area_data, id_vars = '단지코드', value_vars = ['전용면적_10_30', '전용면적_30_40', 
                     '전용면적_40_50', '전용면적_50_100', '전용면적_100이상'])

## column명 변경하기 : 'variable' : '전용면적구간', 'value' : '세대수'
melt_area_data = melt_area_data.rename(columns = {'variable' : '전용면적구간', 'value' : '세대수'})
melt_area_data


# In[149]:


## 기초통계량 확인
## 코드 입력
area_list = ['전용면적_10_30', '전용면적_30_40', 
            '전용면적_40_50', '전용면적_50_100', '전용면적_100이상']
for area_type in area_list:
    print(f'기술통계량 : {area_type}')
    display(melt_area_data.loc[melt_area_data['전용면적구간'] == area_type, ['세대수']].describe().T)
    
## barplot
## 코드 입력
plt.figure(figsize = (10,8))
plt.subplot(2,1,1)
sns.barplot(x = melt_area_data['전용면적구간'], y = melt_area_data['세대수'])
plt.grid()

## boxplot 
## 코드 입력
plt.subplot(2,1,2)
sns.boxplot(x = melt_area_data['전용면적구간'], y = melt_area_data['세대수'])
plt.grid()

plt.show()


# ##### &nbsp; &nbsp; &nbsp; &nbsp; 2-3-2) 건물구분별 면적 비율

# In[38]:


data.columns


# In[40]:


## 단지코드와  건물구분(상가, 아파트)에 관련된 column 추출하기 
## building_data 변수에 저장
## 코드 입력

building_data = data[['단지코드', '상가비율', '아파트비율']]
building_data


# In[44]:


## 건물구분에 관련된 컬럼들을 pd.melt로 재구성하기
## melt_building_data 변수에 저장
## column명 변경하기 : {'variable' : '건물구분', 'value' : '면적비율'}
## 코드 입력

melt_building_data = pd.melt(building_data, id_vars = '단지코드', 
                             value_vars = ['상가비율','아파트비율'])

melt_building_data = melt_building_data.rename(columns = {'variable' : '건물구분', 'value' : '면적비율'})
melt_building_data


# In[87]:


## melt_building_data의 '면적비율'에 대한 기초통계량 확인
## 코드 입력
## 범주별(상가/아파트) '면적비율' 기초 통계량 확인
## 코드 입력

## 기초통계량 확인
## 코드 입력
display(melt_building_data.loc[melt_building_data['건물구분'] == '상가비율'][['면적비율']].describe().T)
display(melt_building_data.loc[melt_building_data['건물구분'] == '아파트비율'][['면적비율']].describe().T)


## barplot
## 코드 입력
plt.figure(figsize = (10,8))
plt.subplot(2,1,1)
sns.barplot(x = melt_building_data['건물구분'], y = melt_building_data['면적비율'])
plt.grid()

## boxplot 
## 코드 입력
plt.subplot(2,1,2)
sns.boxplot(x = melt_building_data['건물구분'], y = melt_building_data['면적비율'])
plt.grid()

plt.show()


# ##### &nbsp; &nbsp; &nbsp; &nbsp; 2-3-3) 공급유형별 면적 비율

# In[49]:


data.columns


# In[56]:


## 단지코드와  공급유형에 관련된 column 추출하기 
## supply_data 변수에 저장
## 코드 입력
supply_data = data[['단지코드', '공급유형_공공임대비율', '공급유형_국민임대비율', 
                         '공급유형_영구임대비율', '공급유형_임대상가비율', '공급유형_장기전세비율', '공급유형_행복주택비율']]
supply_data


# In[57]:


## 건물구분에 관련된 컬럼들을 pd.melt로 재구성하기
## melt_supply_data 변수에 저장
## column명 변경하기 : {'variable' : '공급유형', 'value' : '면적비율'}
## 코드 입력

melt_supply_data = pd.melt(supply_data, id_vars = '단지코드', 
                             value_vars = ['공급유형_공공임대비율', '공급유형_국민임대비율', 
                         '공급유형_영구임대비율', '공급유형_임대상가비율', '공급유형_장기전세비율', '공급유형_행복주택비율'])

melt_supply_data = melt_supply_data.rename(columns = {'variable' : '공급유형', 'value' : '면적비율'})
melt_supply_data


# In[86]:


## 기초통계량 확인
## 코드 입력
supply_list = ['공급유형_공공임대비율', '공급유형_국민임대비율', 
                '공급유형_영구임대비율', '공급유형_임대상가비율', '공급유형_장기전세비율', '공급유형_행복주택비율']
for supply_type in supply_list:
    print(f'기술통계량 : {supply_type}')
    display(melt_supply_data.loc[melt_supply_data['공급유형'] == supply_type, ['면적비율']].describe().T)

##  histplot
## 코드 입력
plt.figure(figsize = (12,10))

plt.subplot(2,1,1)
sns.barplot(x = melt_supply_data['공급유형'], y = melt_supply_data['면적비율'])
plt.grid()

## boxplot
## 코드 입력
plt.subplot(2,1,2)
sns.boxplot(x = melt_supply_data['공급유형'], y = melt_supply_data['면적비율'])
plt.grid()

plt.show()


# #### &nbsp;&nbsp; 2-4) 범주형 데이터 

#    * 분석 방법 : 범주별 빈도수, countplot

# ##### &nbsp; &nbsp; &nbsp; &nbsp; 2-4-1) 지역

# In[84]:


## 범주형 변수의 범주별 빈도수 확인하기 : value_counts()
## 코드 입력
data['지역'].value_counts()

## 그래프 분석하기 : countplot()
## 코드 입력
plt.figure(figsize = (12, 6))
sns.countplot(x = data['지역'])
plt.xticks(rotation = 30)
plt.show()


# ## [정리] 단변량 분석을 통해 파악된 비즈니스 인사이트는 무엇인가요?

# - 전용면적구간 별 평균 세대수는 각각 130세대, 232세대, 165세대, 194세대, 0.06세대 분포
# - 건물 구분 별 아파트 비율이 94%, 상가비율이 6% 정도 분포
# - 공급유형 별 국민임대가 대부분의 비율을 가짐

# <br><br><hr>

# ## 3. 이변량 분석

# - 수치형 feature --> 수치형 target
#     - 전체 변수들 간의 상관관계 구하기
#         * 범주형 변수를 제외한 데이터셋을 이용하여
#         * .corr() + sns.heatmap() 으로 전체 상관계수를 시각화
#     - 상관계수 상위 몇개에 대해서 feature와 target에 대해 
#         * 상관분석을 통해 상관계수가 유의미함을 분석
#         * 산점도를 통해 상관관계를 시각화하여 분석
# - 범주형 feature --> 수치형 target
#     * 범주간에 target의 평균의 차이가 있는지 분석
#     * 범주별 barplot으로 평균의 차이를 시각화하여 분석

# #### &nbsp;&nbsp; 3-1) 수치형 feature --> 수치형 target

# ##### &nbsp; &nbsp; &nbsp; &nbsp; 3-1-1) 전체 수치형 변수들 간의 상관관계 분석

# * 숫자형 데이터의 상호 상관관계

# In[130]:


# 수치형 변수 리스트 정의
## 코드 입력
continous = ['총세대수','공가수','지하철역수','버스정류장수', '등록차량수','전용면적_10_30', '전용면적_30_40', '전용면적_40_50', '전용면적_50_100',
            '전용면적_100이상', '임대보증금', '임대료', '상가비율','아파트비율', '공급유형_공공임대비율', '공급유형_국민임대비율', '공급유형_영구임대비율',
       '공급유형_임대상가비율', '공급유형_장기전세비율', '공급유형_행복주택비율']


# * 숫자형 데이터의 상호 상관관계

# In[92]:


## 데이터 프레임의 상관계수 도출하기 : corr
## 코드 입력
continous_df = data[continous]

continous_df.corr()


# * 상관계수 시각화 

# ## 각 컬럼간 상관계수에 대한 heatmap 그래프 분석
# ## 코드 입력
# plt.figure(figsize = (14,14))
# sns.heatmap(continous_df.corr(), annot = True, vmax = 1, vmin = -1, fmt = '.3f')1.
# plt.show()

# * <span style="color:green"> 질문) Target 등록차량수와 상관 관계가 높은 Feature 컬럼은 무엇인가? (7개 선택)

# - 총세대수 : 0.574
# - 전용면적_50_100 : 0.504351
# - 임대료 : 0.436080
# - 임대보증금 : 0.404836
# - 전용면적_40_50 : 0.369978
# - 공급유형_영구임대비율 : -0.321637	
# - 상가비율, 아파트비율, 공급유형_임대상가비율 : 0.294677 or -0.294677

# ### target과 상관계수 상위 5개 살펴보기

# ##### &nbsp; &nbsp; &nbsp; &nbsp; 3-1-2) 수치형 vs 수치형 (등록차량수) 분석

# * 통계 분석 : 상관분석
# * 그래프 분석 : regplot

# In[ ]:


## 가설 수립
## 귀무 가설(H0) : 
## 대립 가설(H1) :


# In[100]:


# 상위 5개 변수 정의
## 코드 입력
top_5 = ['총세대수', '전용면적_50_100', '임대료', '임대보증금', '전용면적_40_50']


# In[102]:


# target 정의
target = '등록차량수'


# In[108]:


## 코드 입력

for feature in top_5:

    print(f"[{feature}] 통계 분석 및 그래프 분석")

    # 통계 분석 : 통계 분석

    print("***** 통계 분석 *****")
    ## 코드 입력
    result = spst.pearsonr(data[feature], data[target])
    print(feature, " vs ", target, " 상관 분석: ", result)
    
    if result[1] > 0.05:
        print(f"통계분석 결과 : {feature}는 등록차량수에 영향을 주지 않는다")
    else:
        print(f"통계분석 결과 : {feature}는 등록차량수에 영향을 준다")

    # 그래프 분석 : regplot
    print("   ***** 그래프 분석 *****")
    plt.figure(figsize = (12,8))
    sns.regplot(x = data[feature], y = data[target])
    plt.grid()
    plt.show()
    
    
    ## 코드 입력
    
    
    print("")
    print("-"*50)
    


# #### &nbsp;&nbsp; 3-2) 범주형 feature -> 수치형 target

# ##### &nbsp; &nbsp; &nbsp; &nbsp; 3-2-1) 지역

# In[ ]:


## 가설 수립
## 귀무 가설(H0) :
## 대립 가설(H1) : 


# In[110]:


data['지역']


# In[114]:


## 그래프 분석 : barplot
## 코드 입력

plt.figure(figsize = (15,8))

sns.barplot(x = data['지역'], y = data['등록차량수'])
plt.grid()
plt.show()


# In[116]:


# 분산 분석
## 코드 입력

sejong = data.loc[data['지역'] == '세종특별자치시', '등록차량수']
gwangju = data.loc[data['지역']=='광주광역시', '등록차량수']
chungnam = data.loc[data['지역']=='충청남도', '등록차량수']


spst.f_oneway(sejong, gwangju, chungnam)


# ## [정리] 이변량 분석을 통해 파악된 비즈니스 인사이트는 무엇인가요?

# In[ ]:





# ## <font color="orange">**4. 도전 미션** </font> 

# #### &nbsp;&nbsp; target을 '등록차량수_비율'로 바꿔 분석하기

#   - 데이터프레임을 re_data에 복사하고, '등록차량수_비율' 열 추가 : '등록차량수' / '총세대수'
#   - '등록차량수_비율'에 대한 단변량 분석
#   - 전체 수치형 데이터에 대한 상관 관계 분석
#   - Target '등록차량수_비율'에 영향을 미치는 상위 5개 선정 및 이변량 분석
#   - 비즈니스 인사이트 도출

# ##### &nbsp; &nbsp; &nbsp; &nbsp; 4-1) '등록차량수_비율' 열 추가

# In[120]:


## 코드 입력
re_data = data.copy()
re_data['등록차량수_비율'] = re_data['등록차량수'] / re_data['총세대수']
re_data.head(2)


# ##### &nbsp; &nbsp; &nbsp; &nbsp; 4-2) '등록차량수_비율'에 대한 단변량 분석

# In[126]:


## '등록차량수_비율' 변수의 기초통계량 확인
## 코드 입력
display(re_data[['등록차량수_비율']].describe().T)

## 2행 1열 그래프 그리기

## '등록차량수_비율' 변수의 histplot그리기
## 코드 입력
plt.figure(figsize = (10,8))
plt.subplot(2,1,1)
sns.histplot(x = re_data['등록차량수_비율'])
plt.grid()
## '등록차량수' 변수의 boxplot 그리기
## 코드 입력
plt.subplot(2,1,2)
sns.boxplot(x = re_data['등록차량수_비율'])
plt.grid()

plt.show()


# ##### &nbsp; &nbsp; &nbsp; &nbsp; 4-3) 모든 수치형 데이터에 대한 상관관계 분석

# In[135]:


## 코드 입력
continous.append('등록차량수_비율')

# corr()
display(re_data[continous].corr())

# heatmap()
plt.figure(figsize = (18,18))
sns.heatmap(re_data[continous].corr(),  annot = True, vmax = 1, vmin = -1, fmt = '.3f')
plt.show()


# - 상관관계 top5  
# 
# - 등록차량수 : 0.583
# - 임대보증금 : 0.574
# - 공급유형_영구임대비율 : -0.502
# - 임대료 : 0.483
# - 전용면적_10_30 : -0.416
# 

# ##### &nbsp; &nbsp; &nbsp; &nbsp; 4-4) 상위 5개 선정 및 이변량 분석 

# In[136]:


top_5 = ['등록차량수', '임대보증금', '공급유형_영구임대비율', '임대료', '전용면적_10_30']
target = '등록차량수_비율'


# In[139]:


## 코드 입력

for feature in top_5:

    print(f"[{feature}] 통계 분석 및 그래프 분석")

    # 통계 분석 : 통계 분석

    print("***** 통계 분석 *****")
    ## 코드 입력
    result = spst.pearsonr(re_data[feature], re_data[target])
    print(feature, " vs ", target, " 상관 분석: ", result)
    
    if result[1] > 0.05:
        print(f"통계분석 결과 : {feature}는 등록차량수에 영향을 주지 않는다")
    else:
        print(f"통계분석 결과 : {feature}는 등록차량수에 영향을 준다")

    # 그래프 분석 : regplot
    print("   ***** 그래프 분석 *****")
    plt.figure(figsize = (12,8))
    sns.regplot(x = re_data[feature], y = re_data[target])
    plt.grid()
    plt.show()
    
    
    ## 코드 입력
    
    
    print("")
    print("-"*50)
    


# ## [정리] 새로운 Target에 대한 데이터 분석을 통해 파악된 비즈니스 인사이트는 무엇인가요?

# 1. target : 등록차량수
# 
# 상관관계o - 총세대수, 전용면적_5_100, 임대료, 임대보증금
# 
# 2. target : 등록차량수_비율
# 
# 상관관계o - 등록차량수, 임대보증금, 공급유형_영구임대비율, 임대료

# ## 추가해석  
# ### target : 등록차량대수 / 단지내주차면수 

# In[154]:


re_data.shape


# In[178]:


my_data = pd.read_csv('registered_parking_car2.csv')
my_data = my_data[['단지코드', '단지내주차면수']]
my_data = my_data.drop_duplicates(keep = 'first')
my_data


# In[179]:


my_df = pd.merge(re_data, my_data, on = '단지코드', how = 'left')
my_df.shape


# In[180]:


my_df


# In[181]:


my_df['주차과잉면수_비율'] = my_df['등록차량수'] / my_df['단지내주차면수']
my_df


# ## 분석 시각화
# ### '등록차량수_비율'과 '주차과잉면수_비율'간 상관관계

# In[182]:


target = '주차과잉면수_비율'
feature = '등록차량수_비율'

print(f"[{feature}] 통계 분석 및 그래프 분석")

# 통계 분석 : 통계 분석

print("***** 통계 분석 *****")
## 코드 입력
result = spst.pearsonr(my_df[feature], my_df[target])
print(feature, " vs ", target, " 상관 분석: ", result)

if result[1] > 0.05:
    print(f"통계분석 결과 : {feature}는 등록차량수에 영향을 주지 않는다")
else:
    print(f"통계분석 결과 : {feature}는 등록차량수에 영향을 준다")

# 그래프 분석 : regplot
print("   ***** 그래프 분석 *****")
plt.figure(figsize = (12,8))
sns.regplot(x = my_df[feature], y = my_df[target])
plt.grid()
plt.show()


# ## <font color="green"> **Mission Clear** </font> &nbsp; &nbsp; 수고하셨습니다!!

# In[ ]:




