import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import seaborn as sns
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False
train_data = pd.read_csv('data\\train_data.csv')
test_data = pd.read_csv('data\\test_a.csv')
train_data['type'] = 'train'
test_data['type'] = 'test'
# print(train_data.head())
# info：缺失
# print(train_data.info())
# 连续变量的描述信息，包括缺失，中位数，标准差，最小值，最大值，四分位数
# print(train_data.describe())
# 离散型变量的描述信息
# print(train_data.describe(include='0'))
# print(train_data.corr())
# 将训练集标签单独保存
train_label = train_data.pop('tradeMoney')
# 将训练集与测试集合并
all_data = pd.concat([train_data, test_data], ignore_index=True, sort=True)
# print(all_data.info())

# 分别找出分类型和连续型数据
categorical_features = ['rentType', 'houseType', 'houseFloor',  'houseToward',
                       'houseDecoration', 'communityName', 'city', 'region', 'plate', 'buildYear',
                       'tradeTime']
numerical_features = ['ID', 'area', 'totalFloor', 'saleSecHouseNum', 'subwayStationNum', 'busStationNum',
                     'interSchoolNum', 'schoolNum', 'privateSchoolNum', 'hospitalNum', 'drugStoreNum',
                     'gymNum', 'bankNum', 'shopNum', 'parkNum', 'mallNum', 'superMarketNum', 'totalTradeMoney',
                     'totalTradeArea', 'tradeMeanPrice', 'tradeSecNum', 'totalNewTradeMoney', 'totalNewTradeArea',
                     'tradeNewMeanPrice', 'tradeNewNum', 'remainNewNum', 'supplyNewNum', 'supplyLandNum', 'supplyLandArea',
                     'tradeLandNum', 'tradeLandArea', 'landTotalPrice', 'landMeanPrice', 'totalWorkers', 'newWorkers',
                     'residentPopulation', 'pv', 'uv', 'lookNum']
# 缺失值分析
def missing_values(df):
    all_data_na = pd.DataFrame(df.isnull().sum(), columns={'missingNum'})
    all_data_na['existNum'] = len(df) - all_data_na['missingNum']
    all_data_na['sum'] = len(df)
    all_data_na['missingRatio'] = all_data_na['missingNum'] / len(df) * 100
    all_data_na['dtype'] = df.dtypes
    # ascending：默认True升序排列；False降序排列
    all_data_na = all_data_na[all_data_na['missingNum'] > 0].reset_index().sort_values(by=['missingNum', 'index'], ascending=[False, True])
    all_data_na.set_index('index', inplace=True)
    return all_data_na

# print(missing_values(all_data))
# 是否有单调特征列（单调的特征列很大可能是时间）
# 时间列在特征工程的时候，不同的情况下能有很多的变种形式，比如按年月日分箱，或者按不同的维度在时间上聚合分组
def increasing(vals):
    cnt = 0
    len_ = len(vals)
    for i in range(len_ - 1):
        if(vals[i+1] > vals[i]):
            cnt += 1
    return cnt

features = [col for col in train_data.columns]
for feature in features:
    count = increasing(train_data[feature].values)
    if(count / train_data.shape[0] >= 0.5):
        print('单调特征：', feature)
        print('单调特征值个数：', count)
        print('单调特征值比例：', count / train_data.shape[0])

# 对分类型数据进行特征nunique分析
for feature in categorical_features:
    print(feature + "的特征分布如下")
    feature_counts = all_data[feature].value_counts()
    print(feature_counts)
    if (feature not in ['houseType', 'communityName', 'plate', 'buildYear', 'tradeTime']):
        plt.bar(feature_counts.index, height=feature_counts.values)
        plt.title(feature)
        plt.show()

# 删除不需要的数据
# all_data.drop(['city'])

# 统计特征值出现频次大于100的特征
for feature in categorical_features:
    df_value_counts = pd.DataFrame(all_data[feature].value_counts())
    df_value_counts = df_value_counts.reset_index()
    df_value_counts.columns = [feature, 'counts']  # change column names
    print(df_value_counts[df_value_counts['counts'] >= 100])

# Labe 分布
fig, axes = plt.subplots(2, 3, figsize=(20, 5))
fig.set_size_inches(20, 12)
sns.distplot(train_data['tradeMoney'], ax=axes[0][0])
sns.distplot(train_data[(train_data['tradeMoney'] <= 20000)]['tradeMoney'], ax=axes[0][1])
sns.distplot(train_data[(train_data['tradeMoney'] > 20000) & (train_data['tradeMoney'] <= 50000)]['tradeMoney'], ax=axes[0][2])
sns.distplot(train_data[(train_data['tradeMoney'] > 50000) & (train_data['tradeMoney'] <= 100000)]['tradeMoney'], ax=axes[1][0])
sns.distplot(train_data[(train_data['tradeMoney'] > 100000)]['tradeMoney'], ax=axes[1][1])