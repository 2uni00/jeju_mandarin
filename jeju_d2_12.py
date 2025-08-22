# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# -----------------------------
# 0) 폰트 설정
# -----------------------------
def set_font_for_platform():
    try:
        if platform.system() == 'Darwin':
            plt.rc('font', family='AppleGothic')
        elif platform.system() == 'Windows':
            plt.rc('font', family='Malgun Gothic')
        else:
            plt.rc('font', family='NanumGothic')
    except Exception as e:
        print(f"폰트 설정 중 오류 발생: {e}")
    finally:
        plt.rcParams['axes.unicode_minus'] = False

set_font_for_platform()

# -----------------------------
# 1) 데이터 불러오기
# -----------------------------
try:
    df_weather = pd.read_csv('jeju_weather.csv', encoding='utf-8')
    df_mandarin = pd.read_csv('mandarin.csv', encoding='cp949')
except FileNotFoundError:
    print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# -----------------------------
# 2) 기상 데이터 전처리
# -----------------------------
df_weather['측정 일자'] = pd.to_datetime(df_weather['측정 일자'])
df_weather['연도'] = df_weather['측정 일자'].dt.year
df_weather['월'] = df_weather['측정 일자'].dt.month
df_weather['일강수량'] = pd.to_numeric(df_weather['일강수량'], errors='coerce').fillna(0)
df_weather['평균기온'] = pd.to_numeric(df_weather['평균기온'], errors='coerce')
df_weather['최고기온'] = pd.to_numeric(df_weather['최고기온'], errors='coerce')
df_weather['최저기온'] = pd.to_numeric(df_weather['최저기온'], errors='coerce')

# 강수일수 변수 추가 (강수량이 0보다 큰 날을 강수일수로 간주)
df_weather['강수일수'] = df_weather['일강수량'].apply(lambda x: 1 if x > 0 else 0)

monthly_weather = df_weather.groupby(['연도', '월']).agg(
    avg_temp=('평균기온', 'mean'),
    max_temp=('최고기온', 'mean'),
    min_temp=('최저기온', 'mean'),
    total_rain=('일강수량', 'sum'),
    rain_days=('강수일수', 'sum')
).reset_index()

# -----------------------------
# 3) 일조시간 처리 (실제 데이터 기반)
# -----------------------------
file_names = ['제주.csv', '성산.csv', '서귀포.csv', '고산.csv']
df_list = []
for file_name in file_names:
    df = pd.read_csv(file_name, encoding='latin1', sep='\t', skiprows=18, header=None, usecols=[2])
    df.columns = ['all_data']
    df_split = df['all_data'].str.split(',', expand=True)
    df_processed = df_split[[2, 3]]
    df_processed.columns = ['일시', '일조합(hr)']
    df_list.append(df_processed)

# 네 지역 데이터 합치기
df_all_regions = pd.concat(df_list, ignore_index=True)
df_all_regions['일시'] = pd.to_datetime(df_all_regions['일시'], format='%Y-%m', errors='coerce')
df_all_regions['일조합(hr)'] = pd.to_numeric(df_all_regions['일조합(hr)'], errors='coerce')

# 연도-월 기준으로 평균 일조시간 계산
sun_monthly = (
    df_all_regions.dropna(subset=['일시'])
    .groupby([df_all_regions['일시'].dt.year.rename('연도'),
              df_all_regions['일시'].dt.month.rename('월')])['일조합(hr)']
    .mean()
    .reset_index()
)
sun_monthly.columns = ['연도', '월', 'sun_hr']

# 기존 monthly_weather에 병합
monthly_weather = pd.merge(monthly_weather, sun_monthly, on=['연도','월'], how='left')
# -----------------------------
# 4) 감귤 생산량 데이터 전처리
# -----------------------------
df_mandarin['생산량_면적_비율'] = df_mandarin['생산량(톤)'] / df_mandarin['면적(ha)']
df_mandarin_filtered = df_mandarin[(df_mandarin['연도'] >= 2011) & (df_mandarin['연도'] <= 2021)].copy()

# -----------------------------
# 5) 생육 시기별 데이터 집계
# -----------------------------
growth_stages = {
    '화아분화기_꽃눈형성(10~3월)': [10,11,12,1,2,3],
    '개화전(4월)': [4],
    '개화기(5월)': [5],
    '1_2차_생리낙과기(6~7월)': [6,7],
    '과실비대기(8~10월)': [8,9,10]
}

data_list = []
for year in range(2011,2022):
    row = {'연도': year}
    for stage, months in growth_stages.items():
        if stage == '화아분화기_꽃눈형성(10~3월)':
            df_prev = monthly_weather[(monthly_weather['연도']==year-1) & (monthly_weather['월'].isin([10,11,12]))]
            df_curr = monthly_weather[(monthly_weather['연도']==year) & (monthly_weather['월'].isin([1,2,3]))]
            df_stage = pd.concat([df_prev, df_curr])
        else:
            df_stage = monthly_weather[(monthly_weather['연도']==year) & (monthly_weather['월'].isin(months))]
        
        row[f'{stage}_avg_temp'] = df_stage['avg_temp'].mean()
        row[f'{stage}_max_temp'] = df_stage['max_temp'].mean()
        row[f'{stage}_min_temp'] = df_stage['min_temp'].mean()
        row[f'{stage}_total_rain'] = df_stage['total_rain'].sum()
        row[f'{stage}_rain_days'] = df_stage['rain_days'].sum()
        row[f'{stage}_sun_hr'] = df_stage['sun_hr'].mean()
        
    prod = df_mandarin_filtered.loc[df_mandarin_filtered['연도']==year,'생산량_면적_비율']
    row['생산량_면적_비율'] = prod.values[0] if not prod.empty else np.nan
    data_list.append(row)

df_model = pd.DataFrame(data_list).dropna()

# -----------------------------
# 6) 학습용/테스트용 분리
# -----------------------------
X = df_model.drop(columns=['연도','생산량_면적_비율'])
y = df_model['생산량_면적_비율']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 7) 모델 학습
# -----------------------------
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# -----------------------------
# 8) 특성 중요도 분석 및 시각화
# -----------------------------
feature_importances = pd.DataFrame(
    {'feature': X_train.columns, 'importance': rf_model.feature_importances_}
).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Random Forest Feature Importance')
plt.xlabel('중요도')
plt.ylabel('변수')
plt.show()

print("\n--- 특성 중요도 분석 결과 ---")
print(feature_importances)

# -----------------------------
# 9) 최적 조건 탐색 함수 및 규칙 정의
# -----------------------------
external_rules = {
    "화아분화기_꽃눈형성(10~3월)": {"months": [10,11,12,1,2,3], "temp": (5,15), "rain": (150, 400), "sun": (100,180)},
    "개화전(4월)": {"months": [4], "temp": (13,20), "rain": (300, 600), "sun": (150,200)},
    "개화기(5월)": {"months": [5], "temp": (13,20), "rain": (300, 600), "sun": (150,200)},
    "1_2차_생리낙과기(6~7월)": {"months": [6,7], "temp": (22,27), "rain": (500, 1000), "sun": (170,210)},
    "과실비대기(8~10월)": {"months": [8,9,10], "temp": (23,26), "rain": (600, 1200), "sun": (160,210)}
}

def penalty_ratio(val, min_v, max_v):
    # 최적 범위 내: 1.01
    if min_v <= val <= max_v:
        return 1.02
    # 약간 벗어난 경우 (예: 10% 이내): 1.00 ~ 0.99
    elif (min_v * 0.90) <= val < min_v or max_v < val <= (max_v * 1.1):
        return 0.99
    # 중간 정도 벗어난 경우 (예: 20% 이내): 0.98
    elif (min_v * 0.8) <= val < (min_v * 0.90) or (max_v * 1.1) < val <= (max_v * 1.2):
        return 0.98
    # 크게 벗어난 경우: 0.96
    else:
        return 0.97

def adjust_by_external_rules(user_monthly, pred_yield, feature_importances):
    final_penalty = 1.0
    user_monthly_df = pd.DataFrame(user_monthly)
    
    stage_data = {}
    
    df_prev = user_monthly_df[user_monthly_df['월'].isin([10,11,12])].iloc[:3]
    df_curr = user_monthly_df[user_monthly_df['월'].isin([1,2,3])].iloc[:3]
    df_stage_bloom = pd.concat([df_prev, df_curr])
    stage_data['화아분화기_꽃눈형성(10~3월)'] = {
        'avg_temp': df_stage_bloom['avg_temp'].mean(),
        'total_rain': df_stage_bloom['total_rain'].sum(),
        'sun_hr': df_stage_bloom['sun_hr'].mean()
    }
    
    df_stage_preflower = user_monthly_df[user_monthly_df['월'] == 4]
    stage_data['개화전(4월)'] = {
        'avg_temp': df_stage_preflower['avg_temp'].mean(),
        'total_rain': df_stage_preflower['total_rain'].sum(),
        'sun_hr': df_stage_preflower['sun_hr'].mean()
    }
    
    df_stage_flower = user_monthly_df[user_monthly_df['월'] == 5]
    stage_data['개화기(5월)'] = {
        'avg_temp': df_stage_flower['avg_temp'].mean(),
        'total_rain': df_stage_flower['total_rain'].sum(),
        'sun_hr': df_stage_flower['sun_hr'].mean()
    }
    
    df_stage_drop = user_monthly_df[user_monthly_df['월'].isin([6,7])]
    stage_data['1_2차_생리낙과기(6~7월)'] = {
        'avg_temp': df_stage_drop['avg_temp'].mean(),
        'total_rain': df_stage_drop['total_rain'].sum(),
        'sun_hr': df_stage_drop['sun_hr'].mean()
    }
    
    df_stage_growth = user_monthly_df[user_monthly_df['월'].isin([8,9,10])].iloc[-3:]
    stage_data['과실비대기(8~10월)'] = {
        'avg_temp': df_stage_growth['avg_temp'].mean(),
        'total_rain': df_stage_growth['total_rain'].sum(),
        'sun_hr': df_stage_growth['sun_hr'].mean()
    }
    
    importances_dict = feature_importances.set_index('feature')['importance'].to_dict()

    for stage, rule in external_rules.items():
        if stage in stage_data:
            data = stage_data[stage]
            
            temp_importance = importances_dict.get(f'{stage}_avg_temp', 0)
            rain_importance = importances_dict.get(f'{stage}_total_rain', 0)
            sun_importance = importances_dict.get(f'{stage}_sun_hr', 0)
            
            temp_weight = 3
            rain_weight = 1
            sun_weight = 3

            if not pd.isna(data['avg_temp']):
                final_penalty *= (penalty_ratio(data['avg_temp'], *rule['temp']) ** (temp_importance * temp_weight))
            if not pd.isna(data['total_rain']):
                final_penalty *= (penalty_ratio(data['total_rain'], *rule['rain']) ** (rain_importance * rain_weight))
            if not pd.isna(data['sun_hr']):
                final_penalty *= (penalty_ratio(data['sun_hr'], *rule['sun']) ** (sun_importance * sun_weight))
            
    return pred_yield * final_penalty

# -----------------------------
# 10) 사용자 입력 기반 예측
# -----------------------------
print("\n--- 다음년도 감귤 생산량 예측 (월별 입력) ---")
user_monthly_input = []
month_stats = monthly_weather.groupby('월').agg(
    mean_temp=('avg_temp', 'mean'),
    min_temp=('avg_temp', 'min'),
    max_temp=('avg_temp', 'max'),
    mean_rain=('total_rain', 'mean'),
    min_rain=('total_rain', 'min'),
    max_rain=('total_rain', 'max'),
    mean_sun=('sun_hr', 'mean'),
    min_sun=('sun_hr', 'min'),
    max_sun=('sun_hr', 'max'),
    mean_maxtemp=('max_temp', 'mean'),
    mean_mintemp=('min_temp', 'mean'),
    mean_raindays=('rain_days', 'mean')
).reset_index()

monthly_optimal_range = {}
for stage, rule in external_rules.items():
    temp_range = rule['temp']
    rain_range = rule['rain']
    sun_range = rule['sun']
    for month in rule['months']:
        monthly_optimal_range[month] = {
            'temp': temp_range,
            'rain': rain_range,
            'sun': sun_range
        }

def safe_input(prompt, mean_val, min_val, max_val, optimal_range):
    while True:
        try:
            optimal_str = f"{optimal_range[0]}~{optimal_range[1]}"
            val = float(input(f"{prompt} (평균 {mean_val:.2f}, 최적범위 {optimal_str}, 허용범위 {min_val*0.8:.1f}~{max_val*1.2:.1f}): "))
            if min_val*0.8 <= val <= max_val*1.2:
                return val
            else:
                print("⚠️ 잘못된 입력입니다. 다시 입력하세요.")
        except ValueError:
            print("⚠️ 숫자를 입력해 주세요.")

# 사용자로부터 월별 기상 데이터 입력받기
months_to_input = [10, 11, 12] + list(range(1, 11))
for month in months_to_input:
    stats = month_stats[month_stats['월'] == month].iloc[0]
    
    temp_range = monthly_optimal_range[month]['temp']
    rain_range = monthly_optimal_range[month]['rain']
    sun_range = monthly_optimal_range[month]['sun']
    
    temp = safe_input(f"{month}월 평균기온", stats['mean_temp'], stats['min_temp'], stats['max_temp'], temp_range)
    rain = safe_input(f"{month}월 총강수량", stats['mean_rain'], stats['min_rain'], stats['max_rain'], rain_range)
    sun  = safe_input(f"{month}월 평균 일조시간", stats['mean_sun'], stats['min_sun'], stats['max_sun'], sun_range)
    
    max_t = stats['mean_maxtemp']
    min_t = stats['mean_mintemp']
    r_days = stats['mean_raindays']
    
    user_monthly_input.append({'월': month, 
                               'avg_temp': temp, 'max_temp': max_t, 'min_temp': min_t,
                               'total_rain': rain, 'rain_days': r_days, 'sun_hr': sun})

# 사용자 입력을 기반으로 한 모델 입력 데이터 구성
user_input = {}
user_monthly_df = pd.DataFrame(user_monthly_input)

for stage, months in growth_stages.items():
    if stage == '화아분화기_꽃눈형성(10~3월)':
        df_prev = user_monthly_df[user_monthly_df['월'].isin([10,11,12])].iloc[:3]
        df_curr = user_monthly_df[user_monthly_df['월'].isin([1,2,3])].iloc[:3]
        df_stage = pd.concat([df_prev, df_curr])
    elif stage == '과실비대기(8~10월)':
        df_stage = user_monthly_df[user_monthly_df['월'].isin(months)].iloc[-3:]
    else:
        df_stage = user_monthly_df[user_monthly_df['월'].isin(months)]
        
    user_input[f'{stage}_avg_temp'] = df_stage['avg_temp'].mean()
    user_input[f'{stage}_max_temp'] = df_stage['max_temp'].mean()
    user_input[f'{stage}_min_temp'] = df_stage['min_temp'].mean()
    user_input[f'{stage}_total_rain'] = df_stage['total_rain'].sum()
    user_input[f'{stage}_rain_days'] = df_stage['rain_days'].sum()
    user_input[f'{stage}_sun_hr'] = df_stage['sun_hr'].mean()

X_new = pd.DataFrame([user_input])
X_new = X_new[X_train.columns]

# 최적 조건 예상 생산량 계산
optimal_conditions_dict = {}
for stage, rule in external_rules.items():
    optimal_conditions_dict[f"{stage}_avg_temp"] = np.mean(rule['temp'])
    optimal_conditions_dict[f"{stage}_total_rain"] = np.mean(rule['rain'])
    optimal_conditions_dict[f"{stage}_sun_hr"] = np.mean(rule['sun'])

for col in X_train.columns:
    if col not in optimal_conditions_dict:
        optimal_conditions_dict[col] = X_train[col].mean()

X_optimal = pd.DataFrame([optimal_conditions_dict])
X_optimal = X_optimal[X_train.columns]

optimal_yield = adjust_by_external_rules(user_monthly_input, rf_model.predict(X_optimal)[0], feature_importances)
print(f"최적 조건 예상 생산량: {optimal_yield:.2f} 톤/ha")

# 최종 예측 결과 계산 및 출력
predicted_yield = rf_model.predict(X_new)[0]
final_yield = adjust_by_external_rules(user_monthly_input, predicted_yield, feature_importances)
ratio_to_optimal = final_yield / optimal_yield * 100

print(f"\n모델 예측 단위 면적당 생산량: {predicted_yield:.2f} 톤/ha")
print(f"외부 기준 보정 후 생산량: {final_yield:.2f} 톤/ha")
print(f"최적 조건 대비 {ratio_to_optimal:.1f}% 수준")