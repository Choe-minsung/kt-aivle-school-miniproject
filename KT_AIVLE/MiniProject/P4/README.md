

# Mini Project 4

### subject : 1:1문의 텍스트 분류 [LSTM] & 차량 Object Detection [Yolov5]

- duration : 2023.10.16 ~ 2023.10.20
- stack : Python (Jupyter Notebook, Colab)

#### pipeline
- Text Classification  
1. 데이터전처리 : 개행제거, 특수문자제거, 단어분리, 불용어 제거, 한글자 제거
2. 데이터탐색 : 단어갯수측정, padding 갯수지정
3. 단어사전 만들기 : 빈도 수 측정, 빈도 수 합계(비율) 측정, 단어사전 크기 지정, 토큰화(단어 인덱스화)
4. feature padding 적용(pad_sequences), target 정수인코딩
5. model 생성(SimpleRNN, LSTM) 및 검증

<img src='https://github.com/Choe-minsung/Project/blob/30697abd83ebf9ce68347d0470adde29653ed337/KT_AIVLE/MiniProject/P4/WC.png' width='700'/>

- Object Detection
1. Video → Image 추출 (OpenCV)
2. Roboflow에 Car Labeled dataset 가져오기, yaml parsing
3. Yolov5n ~ Yolov5x 학습

<img src='https://github.com/Choe-minsung/Project/blob/30697abd83ebf9ce68347d0470adde29653ed337/KT_AIVLE/MiniProject/P4/OD.jpg' width='500'/>
