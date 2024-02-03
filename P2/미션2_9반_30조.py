#!/usr/bin/env python
# coding: utf-8

# # **Mission 2_주차 등록 수요 데이터 전처리**

# ## <미션>

#  1) [단지별 공통 정보]와 [단지 상세 정보]를 분리하기
#  2) 범주형 변수의 category 수 줄이기
#  3) [단지별 공통 정보]의 중복행 제거
#  4) [단지 상세 정보] 집계를 통해 단지별 정보 구하기 
#       * 전용면적 구간별 총 세대수
#       * 단지별 임대보증금, 임대료
#       * 임대건물구분 비율 (면적 비율)
#       * 공급 유형 비율 (면적 비율)
#  5) [단지별 공통 정보]와 [단지 상세 정보]의 집계 내용을 합치기 

# In[ ]:





# ## <환경설정>

# ### &nbsp;&nbsp; 1) 라이브러리 불러오기

# * 기본적으로 필요한 라이브러리를 import 하도록 코드가 작성되어 있습니다.
# * 필요하다고 판단되는 라이브러리를 추가하세요.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 아래 필요한 라이브러리, 함수를 추가하시오.


# ### &nbsp;&nbsp; 2) 한글 폰트 설정하기

# In[2]:


plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False


# ### &nbsp;&nbsp; 3) 데이터 불러오기

# * 파일명 : registered_parking_car.csv
# * data 변수에 저장하기 

# In[3]:


## 코드 입력
data = pd.read_csv('registered_parking_car.csv')
data.head()


# <br><br><hr>

# ## 1. 기본정보 확인하기
# 

# * **세부 요구사항**
#     - 불러온 데이터의 형태, 기초통계량, 정보 등을 확인합니다.
#     - 특히 .info() 를 통해서 각 변수별 데이터타입이 적절한지 확안합니다.

# ### &nbsp;&nbsp; 1-1) 전체 데이터의 행, 열 개수 확인

# In[4]:


## shape
## 코드 입력
data.shape


# ### &nbsp;&nbsp; 1-2) 전체 데이터의 상위 5개 행 확인

# In[5]:


## head
## 코드 입력
data.head()


# ### &nbsp;&nbsp; 1-3) 전체 데이터의 모든 변수명 (columns) 확인

# In[6]:


## columns
## 코드 입력
data.columns


# ### &nbsp;&nbsp; 1-4) 결측치 (N/A) 존재 여부 확인, 각 컬럼의 데이터 타입 확인

# In[7]:


data.isna().sum()


# In[8]:


## info, isna
## 코드 입력
data.info()


# <br><br><hr>

# ## 2. [단지별 공통 정보]와 [단지 상세 정보] 분리 및 중복 제거

# ### &nbsp;&nbsp; 3-1) 공통정보/상세정보 분리

#  * [단지별 공통 정보] 
#     * 대상 컬럼 : 단지코드, 총 세대수, 지역, 공가수, 도보 10분거리 내 지하철역 수(환승노선 수 반영), 도보 10분거리 내 버스정류장 수, 등록 차량수
#     * 변수명 : danji_main
#  * [단지 상세 정보] 
#     * 대상 컬럼 : 단지 코드, 임대건물구분, 공급유형, 전용면적, 전용면적별세대수, 임대보증금, 임대료
#     * 변수명 : danji_detail
#   

# In[9]:


## 단지별 공통 정보 
## 코드 입력
danji_main = data[['단지코드', '총세대수', '지역', '공가수',
                  '도보 10분거리 내 지하철역 수(환승노선 수 반영)',
                  '도보 10분거리 내 버스정류장 수', '등록차량수']]

## 단지 상세 정보
## 코드 입력
danji_detail = data[['단지코드', '임대건물구분', '공급유형',
                   '전용면적', '전용면적별세대수', '임대보증금', '임대료']]


# ### &nbsp;&nbsp; 3-2) 전체 데이터의 모든 변수명 (columns) 확인 긴 글자로 된 column명 변경하기

#  -  '도보 10분거리 내 지하철역 수(환승노선 수 반영)'  ==> '지하철역수'
#  - '도보 10분거리 내 버스정류장 수'  ==> '버스정류장수'

# In[10]:


##  컬럼 변경 하기 : rename
## 코드 입력
danji_main = danji_main.rename(columns = {'도보 10분거리 내 지하철역 수(환승노선 수 반영)' : '지하철역수',
                                         '도보 10분거리 내 버스정류장 수' : '버스정류장수'})



# ### &nbsp;&nbsp; 3-3) [단지별 공통 정보]의 중복행 제거하기

# In[11]:


## 중복행 제거 : drop_duplicates
## 코드 입력
danji_main = danji_main.drop_duplicates(keep = 'first')
danji_main


# In[12]:


# 단지 코드별 행 개수를 체크하여 중복 제거가 잘 되었는지 확인  : groupby 활용
# groupby로 count한 값이 1보다 큰 출력값은 중복임


## '단지코드'별로 그룹핑하여 '총세대수' 열의 개수를 세기 (count)
## unique_check 변수에 저장하기
## 코드 입력
unique_check = danji_main.groupby(danji_main['단지코드'], as_index = False)['총세대수'].count()

# unique_check 데이터프레임의 컬럼명을 ['단지코드','count']로 컬럼 이름 바꾸기 
## 코드 입력
unique_check.columns = ['단지코드', 'count']

# unique_check 데이터프레임 중 'count'열이 1보다 큰 경우를 조회하기
## 코드 입력
unique_check.loc[unique_check['count'] > 1,:]


# ## 4. 범주형 변수의 category 수 줄이기, 숫자형 변수 확인하기 

# ### &nbsp;&nbsp; 4-1) ‘공급유형’: ‘공공임대(10년)’, ‘공공임대(5년)’…등으로 나뉜 것을 하나의 범주 값인 ‘공공임대’ 로 통합하기

# In[13]:


## [단지 상세 정보] '공급유형'의 category 확인하기 : value_counts
## 코드 입력
danji_detail['공급유형'].value_counts()


# In[14]:


## ['공공임대(10년)', '공공임대(50년)', '공공임대(5년)', '공공임대(분납)'] ==> '공공임대'로 수정
## 코드 입력
danji_detail.loc[danji_detail['공급유형'].isin(['공공임대(10년)', '공공임대(50년)', '공공임대(5년)', '공공임대(분납)']),'공급유형'] = '공공임대'

## 확인 : value_counts()
## 코드 입력
danji_detail['공급유형'].value_counts()


# ### &nbsp;&nbsp; 4-2) 숫자가 입력되어야 하는 컬럼에 문자 ‘-’가 입력된 경우를 찾아, 숫자로 변환하기

# * 대상 컬럼 찾기 
# * object 타입을 int64나 float64로 변환

# In[15]:


## [단지 상세 정보]에서 숫자형이어야 하는데 object형으로 보이는 컬럼은 무엇인가요? : info()
## 코드 입력
danji_detail.info()
danji_detail # 임대보증금, 임대료가 object형!


# In[16]:


## 숫자가 아닌 데이터를 찾고 있는 행 찾기
## 코드 입력
danji_detail.loc[danji_detail['임대보증금'].str.isnumeric() == False,:]

# 즉, 대부분 숫자지만 '-' 문자열을 가지고있는 행이 존재함


# In[17]:


## 숫자가 아닌 데이터를 찾고 있는 행 찾기
## 코드 입력
danji_detail.loc[danji_detail['임대료'].str.isnumeric() == False,:]

# 즉, 대부분 숫자지만 '-' 문자열을 가지고있는 행이 존재함


# In[18]:


## 숫자가 아닌 데이터가 있는 열을 0으로 대체하기

## loc를 활용하는 경우
## 코드 입력
danji_detail.loc[danji_detail['임대보증금'].str.isnumeric() == False, '임대보증금'] = 0
danji_detail.loc[danji_detail['임대료'].str.isnumeric() == False, '임대료'] = 0
danji_detail

## replace를 활용하는 경우
## 코드 입력
# danji_detail['임대보증금'].replace(danji_detail['임대보증금'].str.isnumeric() == False, 0)
# danji_detail['임대료'].replace(danji_detail['임대료'].str.isnumeric() == False, 0)


# In[19]:


## 해당 칼럼에  데이터 중 '-' 가 있는지 확인하기
## 코드 입력

danji_detail.loc[danji_detail['임대보증금'] == '-', :]


# In[20]:


danji_detail.loc[danji_detail['임대료'] == '-', :]


# In[21]:


## 해당 칼럼의 dtype을 object --> float로 수정하기 : astype
## 코드 입력
danji_detail[['임대보증금', '임대료']] = danji_detail[['임대보증금', '임대료']].astype(float)

## 확인하기 : info
## 코드 입력
danji_detail.info()


# <br><br><hr>

# ## 5. [단지 상세 정보] 집계를 통해 단지별 정보 구하기

# ### &nbsp;&nbsp; 5-1) 전용면적 구간별 총 세대수 구하기

# * [단지 상세 정보]에서 단지코드, 전용면적, 전용면적별세대수 만을 추출하여 base_5_1 에 저장하기
# * 전용면적 구간별 세대 수 집계 (groupby)
# * 단지 코드를 index로, 전용면적 구간을 컬럼(열)으로 하여 전용면적별세대수 구하기 (pivot)

# In[22]:


## [단지 상세 정보]에서 단지코드, 전용면적, 전용면적별세대수 만을 추출하여 base_5_1 에 저장하기
## 코드 입력
base_5_1 = danji_detail[['단지코드', '전용면적', '전용면적별세대수']]
base_5_1.tail(2)


# In[23]:


# base_5_1['전용면적'] 데이터의 기초 통계량 확인
## 코드 입력
print(base_5_1['전용면적'].describe())

# base_5_1['전용면적]' 데이터 분포 시각화 확인 (sns.histplot)
## 코드 입력
sns.histplot(base_5_1['전용면적'])
plt.show()


# In[24]:


## 전용 면적을 의미있는 구분할 수 있는 구간 나눠보고, 그에 맞는 라벨 설정하기
## 코드 입력
bins = [-np.inf, 32, 40, 51, np.inf]
labels = ['0_32', '32_40', '40_51', '51_']
## base_5_1 '전용면적'을 정해진 bins/labels 기준으로 나누고, '전용면적구간'이름으로 추가하기
## 코드 입력
base_5_1['전용면적구간'] = pd.cut(base_5_1['전용면적'], bins = bins, labels = labels)
base_5_1


# In[25]:


## 전용면적 구간별 세대수 집계하기 : groupby
## 결과를 group_5_1에 저장하기
## 코드 입력
group_5_1 = base_5_1.groupby(by = ['단지코드','전용면적구간'], as_index = False)['전용면적별세대수'].sum()

## group_5_1 값 확인하기
## 코드 입력
group_5_1.tail(10)


# In[26]:


## 단지 코드를 index로, 전용면적 구간을 컬럼(열)으로 하여 전용면적별세대수 구하기 (pivot)
## 결과를 result_5_1 저장
## 단지 코드를 index --> 컬럼으로 변경하기 : reset_index, drop=False, inplace=True
## 코드 입력
result_5_1 = group_5_1.pivot(index = '단지코드', columns = '전용면적구간', values = '전용면적별세대수')

result_5_1 = result_5_1.reset_index(drop=False)
result_5_1.head(2)


# ### &nbsp;&nbsp; 5-2) 단지별 임대보증금, 임대료 구하기 (평균, 중앙값)

# * 단지별 임대보증금, 임대료의 전체 평균/중앙값 구하기

# In[27]:


danji_detail


# In[28]:


## [단지 상세 정보]에서 단지코드, 임대보증금, 임대료 정보 추출하기 : base_5_2 에 저장
## 코드 입력
base_5_2 = danji_detail[['단지코드', '임대보증금', '임대료']]
base_5_2


# In[29]:


## 단지별 임대보증금, 임대료 평균값 구하기 : groupby, mean
## group_5_2_mean 저장
## 코드 입력
group_5_2_mean = base_5_2.groupby(base_5_2['단지코드'], as_index = False)[['임대보증금', '임대료']].mean()
group_5_2_mean


# In[30]:


## 단지별 임대보증금, 임대료 중앙값 구하기  : groupby, median
## group_5_2_median 저장
## 코드 입력
group_5_2_median = base_5_2.groupby(base_5_2['단지코드'], as_index = False)[['임대보증금', '임대료']].median()
group_5_2_median



# ### &nbsp;&nbsp; <font color="orange">**[도전 미션]** </font>  단지별 임대보증금, 임대료의 가중 평균 구하기

#    * 1) 임대보증금 * 세대수, 임대료 * 세대수 구하기 (전용면적별 총 임대보증금, 총 임대료)
#    * 2) 단지별 총 임대보증금, 총 임대료, 총 세대수 구하기 (groupby)
#    * 3) 임대보증금 가중 평균 = 총 임대보증금 / 총 세대수, 임대료 가중 평균 = 총 임대료 / 총 세대수

# In[31]:


danji_detail.head(2)


# In[32]:


base_5_1.head(2)


# In[33]:


base_5_2.head(2)


# In[34]:


group_5_1.head(2)


# In[35]:


danji_detail.head()


# In[36]:


## [임대보증금, 임대료]  * 전용면적별세대수 ==> [세대수X임대보증금, 세대수X임대료] 구하기
## 코드 입력
danji_detail['세대수X임대보증금'] = danji_detail['전용면적별세대수'] * danji_detail['임대보증금']
danji_detail['세대수X임대료'] = danji_detail['전용면적별세대수'] * danji_detail['임대료']

## 단지별 [세대수X임대보증금, total_임대료] 합계 구하기 : groupby
## group_5_2_weighted_mean 에 저장
## 코드 입력
group_5_2_weighted_mean = danji_detail.groupby(danji_detail['단지코드'], as_index = False)['전용면적별세대수', '세대수X임대보증금', '세대수X임대료'].sum()


## 단지별 [total_임대보증금, total_임대료] 합계를 단지별 총 전용면적별세대수로 나눠 가중 평균 구하기
## 코드 입력
group_5_2_weighted_mean['임대보증금_가중평균'] = group_5_2_weighted_mean['세대수X임대보증금'] / group_5_2_weighted_mean['전용면적별세대수']
group_5_2_weighted_mean['임대료_가중평균'] = group_5_2_weighted_mean['세대수X임대료'] / group_5_2_weighted_mean['전용면적별세대수']

group_5_2_weighted_mean


# * 평균/중앙값/가중 평균 중 대표값 선정하기
# *  <font color='red'> **도전 미션을 하지 않아도 평균/중앙값 중에서 대표값 선정해야 함!!!!! [필수]** </font>

# In[37]:


## danji_main 데이터에 합칠 정보는 무엇인가?
## 1) 평균     2) 중앙값    3) 가중평균
## 선택된 정보를 result_5_2 저장하기
## 코드 입력

# 1)평균으로 사용시
result_5_2 = group_5_2_weighted_mean
result_5_2.shape


# ### &nbsp;&nbsp; <font color="orange">**[도전 미션]** </font>  5-3) 임대건물구분 비율 구하기

# * 임대건물구분 비율, 공급유형 비율 집계에 필요한 정보만 추출
# * 전용면적에 세대수를 반영하여 총면적 열 추가

# In[38]:


## [단지 상세 정보]의 ['단지코드', '임대건물구분', '공급유형', '전용면적', '전용면적별세대수'] 열을 base_5_3 변수에 저장
## 코드 입력
base_5_3 = danji_detail[['단지코드', '임대건물구분', '공급유형', '전용면적', '전용면적별세대수']]

## 전용면적 * 전용면적별세대수 구하기 (열 이름 : 세대수X전용면적)
## 코드 입력
base_5_3['세대수X전용면적'] = base_5_3['전용면적'] * base_5_3['전용면적별세대수']
base_5_3.head(2)


# In[70]:


base_5_3.loc[(base_5_3['단지코드'] == 'C1206') & (base_5_3['임대건물구분'] == '상가'),:]


# * 임대건물구분별 면적 비율
#     * 단지별 임대건물구분(상가,아파트)별 총 면적 계산
#     * 비율로 변환
# 

# In[39]:


## 단지코드와 임대건물구분으로 '세대수X전용면적' 집계하기
## group_5_3에 저장
## 코드 입력
group_5_3 = base_5_3.groupby(by = ['단지코드', '임대건물구분'], as_index = False)['세대수X전용면적'].sum()
group_5_3


# In[40]:


## 단지코드를 index로 임대건물구분을 열로, 총 면적을 값으로 pivot table 구하기
## pivot_5_3 에 저장
## 코드 입력
pivot_5_3 = group_5_3.pivot(index = '단지코드', columns = '임대건물구분', values = '세대수X전용면적')

## index로 적용된 단지코드를 열로 변경 : reset_index
## 코드 입력
pivot_5_3 = pivot_5_3.reset_index(drop=False)
pivot_5_3.head(2)


# In[41]:


## pivlot table의 NaN값으로 0으로 대체하기
## 코드 입력
pivot_5_3 = pivot_5_3.fillna(0)
pivot_5_3


# In[60]:


## pivlot table에서 단지별 총 면적 구하기 : 상가면적 + 아파트 면적
## 코드 입력
pivot_5_3['총면적'] = pivot_5_3['상가'] + pivot_5_3['아파트']

## pivlot table에서 상가비율, 아파트비율 구하기 : 상가면적 / (상가+아파트), 아파트면적 / (상가+아파트)
## 코드 입력
pivot_5_3['상가비율'] = pivot_5_3['상가'] / pivot_5_3['총면적']
pivot_5_3['아파트비율'] = pivot_5_3['아파트'] / pivot_5_3['총면적']

## pivot table의 ['단지코드', '상가비율', '아파트비율'] 정보를 result_5_3에 저장하기
## 코드 입력
result_5_3 = pivot_5_3[['단지코드', '상가비율', '아파트비율']]
result_5_3.head(2)


# In[62]:


# 상가비율과 아파트비율이 섞여있는 행 출력
result_5_3.loc[(result_5_3['상가비율'] != 1.0) & (result_5_3['상가비율'] != 0.0)].head()


# In[58]:


result_5_3.tail(10)


# ### &nbsp;&nbsp; <font color="orange">**[도전 미션]** </font>  5-4) 공급유형 비율 구하기

# * 5-4) 에서 저장된 base_5_3 변수를 이용하여 집계
# 

# In[43]:


base_5_3.head(2)


# In[44]:


## 단지코드와 공급유형별로 총면적 집계하기
## group_5_4 저장
## 코드 입력
group_5_4 = base_5_3.groupby(by = ['단지코드', '공급유형'], as_index = False)['세대수X전용면적'].sum()
group_5_4


# In[45]:


## 단지코드를 index로 공급유형을 열로, '세대수X전용면적'을 값으로 pivot table 구하기
## pivot_5_4 저장
## 코드 입력
pivot_5_4 = group_5_4.pivot(index = '단지코드', columns = '공급유형', values = '세대수X전용면적')

# index에 적용된 단지코드를 컬럼으로 변경하기 : reset_index
## 코드 입력
pivot_5_4 = pivot_5_4.reset_index(drop = False)
pivot_5_4


# In[46]:


## pivlot table의 NaN값으로 0으로 대체하기
## 코드 입력
pivot_5_4 = pivot_5_4.fillna(0)
pivot_5_4


# In[47]:


## pivot table의 공급유형별 면적을 모두 더해 '총면적' 구하기
## 코드 입력
pivot_5_4['총면적'] = pivot_5_4['공공임대'] + pivot_5_4['국민임대'] + pivot_5_4['영구임대'] + pivot_5_4['임대상가'] + pivot_5_4['장기전세'] + pivot_5_4['행복주택']

## pivot_5_4에서 공규유형별 면적 비율 구하기 : 각 면적 / 총면적
## 코드 입력
pivot_5_4['공공임대비율'] = pivot_5_4['공공임대'] / pivot_5_4['총면적']
pivot_5_4['국민임대비율'] = pivot_5_4['국민임대'] / pivot_5_4['총면적']
pivot_5_4['영구임대비율'] = pivot_5_4['영구임대'] / pivot_5_4['총면적']
pivot_5_4['임대상가비율'] = pivot_5_4['임대상가'] / pivot_5_4['총면적']
pivot_5_4['장기전세비율'] = pivot_5_4['장기전세'] / pivot_5_4['총면적']
pivot_5_4['행복주택비율'] = pivot_5_4['행복주택'] / pivot_5_4['총면적']

## 단지 코드의 공급유형별 면적 비율을 result_5_4에 저장하기 
## 코드 입력
result_5_4 = pivot_5_4.loc[:,['단지코드', '공공임대비율', '국민임대비율', '영구임대비율', '임대상가비율',
                            '장기전세비율', '행복주택비율']]
result_5_4


# <br><br><hr>

# ## 6. [단지별 공통 정보]에 [단지 상세 정보] 집계 내용을 합치기 

# * [단지별 공통 정보] : **danji_main**
# * [단지 상세 정보] 집계
#    * 전용 면적 구간별 세대수 : **result_5_1**
#    * 임대보증금/임대료 : **result_5_2
#    * 임대건물구분별 면적 비율 : **result_5_3**
#    * 공급유형별 면적 비율 : **result_5_4**
# * 합치기 :  merge를 사용할 때, **how = 'left', on = '단지코드'** 옵션 이용

# ### 6-1) 단지별 공통 정보 + 전용면적 구간별 세대수

# In[48]:


## result 변수에 저장하기
## 코드 입력

result = pd.merge(danji_main, result_5_1, how = 'left', on = '단지코드')
result


# ### 6-2)  result + 임대보증금/임대료

# In[49]:


## 코드 입력
result = pd.merge(result, result_5_2, how = 'left', on = '단지코드')
result


# ### 6-3) result + 임대건물구분별 면적 비율 (도전 미션 완료 시)

# In[50]:


## 코드 입력
result = pd.merge(result, result_5_3, how = 'left', on = '단지코드')
result


# ### 6-4) 공급유형별 면적 비율 (도전 미션 완료 시)

# In[51]:


## 코드 입력
result = pd.merge(result, result_5_4, how = 'left', on = '단지코드')



# In[52]:


result


# ##  7. 데이터셋 저장하기

# * registerd_parking_preprocessed.csv 파일로 저장하기

# In[54]:


## 코드 입력
result.to_csv('registerd_parking_preprocessed.csv', index = False)


# ## <font color="green"> **Mission Clear** </font> &nbsp; &nbsp; 수고하셨습니다!!
