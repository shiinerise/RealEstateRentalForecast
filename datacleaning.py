import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# print(all_data.info())

# 分别找出分类型和连续型数据
categorical_features = ['rentType', 'houseType', 'houseFloor',  'houseToward',
                       'houseDecoration', 'communityName', 'region', 'plate', 'buildYear',
                       'tradeTime']
numerical_features = ['ID', 'area', 'totalFloor', 'saleSecHouseNum', 'subwayStationNum', 'busStationNum',
                     'interSchoolNum', 'schoolNum', 'privateSchoolNum', 'hospitalNum', 'drugStoreNum',
                     'gymNum', 'bankNum', 'shopNum', 'parkNum', 'mallNum', 'superMarketNum', 'totalTradeMoney',
                     'totalTradeArea', 'tradeMeanPrice', 'tradeSecNum', 'totalNewTradeMoney', 'totalNewTradeArea',
                     'tradeNewMeanPrice', 'tradeNewNum', 'remainNewNum', 'supplyNewNum', 'supplyLandNum', 'supplyLandArea',
                     'tradeLandNum', 'tradeLandArea', 'landTotalPrice', 'landMeanPrice', 'totalWorkers', 'newWorkers',
                     'residentPopulation', 'pv', 'uv', 'lookNum']

def preprocessingData(data):
    # 缺失值处理
    def data_processing(data):
        data[data['rentType'] == '--']['rentType'] = '未知方式'

    columns = ['rentType', 'houseType', 'houseFloor', 'houseToward',
               'houseDecoration', 'communityName', 'region', 'plate']
    for feature in columns:
        data[feature] = LabelEncoder().fit_transform(data[feature])

    # 将buildYear转换成整形数据
    # print(data['buildYear'][data['buildYear'] != '暂无信息'].mode())
    buildYearMode = pd.DataFrame(data['buildYear'][data['buildYear'] != '暂无信息'].mode())
    # print(data[data['buildYear'] == '暂无信息'].index)
    data.loc[data[data['buildYear'] == '暂无信息'].index, 'buildYear'] = buildYearMode.iloc[0, 0]
    data['buildYear'] = data['buildYear'].astype('int')

    # 处理pv和uv的空值，用平均值填充
    data['pv'].fillna(data['pv'].mean(), inplace=True)
    data['uv'].fillna(data['uv'].mean(), inplace=True)
    data['pv'] = data['pv'].astype(int)  # 为什么要把float转化成int？
    data['uv'] = data['uv'].astype(int)

    # 分割交易时间
    def month(x):
        return int(x.split('/')[1])
    def day(x):
        return int(x.split('/')[2])
    data['month'] = data['tradeTime'].apply(lambda x: month(x))
    data['day'] = data['tradeTime'].apply(lambda x: day(x))
    # print(data['month'])

    # 去掉不需要的特征
    data.drop(['ID', 'city', 'tradeTime'], axis=1, inplace=True)
    return data

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    mpl.rcParams['axes.unicode_minus'] = False
    train_data = pd.read_csv('data\\train_data.csv')
    test_data = pd.read_csv('data\\test_a.csv')
    train_data['type'] = 'train'
    test_data['type'] = 'test'
    all_data = pd.concat([train_data, test_data], ignore_index=True, sort=True)
    preprocessingData(all_data)