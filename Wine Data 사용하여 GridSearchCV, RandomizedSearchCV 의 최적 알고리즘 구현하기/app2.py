import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV

def trans (joblib_model, Model_type):
    wine_dict = {"Red Wine 🍷" : 0, "White Wine 🍾" : 1}
    reverse_mapping_dict = {value: key for key, value in wine_dict.items()}
    prediction = joblib_model.predict(features)
    st.success(f"{Model_type} 모델은 {reverse_mapping_dict[prediction[0]]}이라고 생각합니다.")
    return

# 모델별 joblib 파일 경
filename_GridSearchCV = "/content/Wine_GridSearchCV.joblib"
filename_RandomizedSearchCV = "/content/Wine_RandomizedSearchCV.joblib"

# joblit을 사용하여 모델 불러오기
joblib_model_GridSearchCV = None
joblib_model_RandomizedSearchCV = None

# 모델 읽어오기
try:
    with open(filename_GridSearchCV, 'rb') as file:         # GridSearchCV 모델
        joblib_model_GridSearchCV = joblib.load(file)
    print(f"모델이 '{filename_GridSearchCV}' 파일에서 성공적으로 불러와졌습니다 (joblib).")
    with open(filename_RandomizedSearchCV, 'rb') as file:   # RandomizedSearchCV 모델
        joblib_model_RandomizedSearchCV = joblib.load(file)
    print(f"모델이 '{filename_RandomizedSearchCV}' 파일에서 성공적으로 불러와졌습니다 (joblib).")
except FileNotFoundError as e:
    print(f"모델이 파일 에러: ", e)

st.header(":rainbow[Wine 데이터를 DecisionTree 학습]")

alcol = st.number_input("알코올 농도", 0.0, 15.0, 5.0)
sugar = st.number_input("당도", 0.0, 100.0, 5.0)
ph = st.number_input("pH", 0.0, 5.0, 0.0)

if st.button("Siuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") : 
    features = [[alcol, sugar, ph]]
    col = st.columns(2)
    with col[0]:
        trans(joblib_model_GridSearchCV,"GridSearchCV")
    with col[1]:
        trans(joblib_model_RandomizedSearchCV, "RandomizedSearchCV")