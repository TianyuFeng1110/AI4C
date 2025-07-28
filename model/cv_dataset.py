import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

# 加载和构建预测夜间灯光强度数据集的预处理函数
def excel_predict_light_filter(file_name):
    df = pd.read_excel(file_name)

    # 删除'平均地表温度'列中值为0的所有行(这种情况说明该值缺失，上海不可能存在年平均地表温度为0的地方)
    df = df[df['lst_day_c'] != 0]

    # 将第一行的列名以及坐标信息作为返回
    ids = df['FID'].values
    coordinates = df[['POINT_X', 'POINT_Y']].values
    features = df.drop(columns=['FID', 'POINT_X', 'POINT_Y', 'light_dnb_', 'lst_day_c', 'lst_night_', 'bk_st_wk',
                                'bk_st_we', 'bk_ar_we', 'bk_str_wk', 'bk_str_we']).values  # 除去宜居环境变量的其他列作为模型输入的特征
    labels = df[['light_dnb_', 'lst_day_c', 'lst_night_', 'bk_st_wk', 'bk_st_we', 'bk_ar_we', 'bk_str_wk', 'bk_str_we']].values  # 所有的宜居环境变量列作为标签(要预测的y)
    return features, labels, ids, coordinates

    
# 构建 CityTransDataSet
class CityTransDataSet(Dataset):
    def __init__(self, features, labels, ids, coordinates):
        self.features = features.clone().detach().float()
        self.labels = labels.clone().detach().float()
        self.ids = ids.clone().detach().float()
        self.coordinates = coordinates.clone().detach().float()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.ids[index], self.coordinates[index]
    


def get_five_fold_datasets(data_file, random_seed=42):
    # 加载并处理数据
    features, labels, ids, coordinates = excel_predict_light_filter(data_file)
    
    # 将数据转为 tensor
    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.float)

    # 创建 CityTransDataset
    ids = torch.tensor(ids, dtype=torch.int).unsqueeze(1)
    coordinates = torch.tensor(coordinates, dtype=torch.float)
    city_trans_dataset = CityTransDataSet(features, labels, ids, coordinates)
    
    # 使用 KFold 替代 StratifiedKFold
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_seed)  # 五折交叉验证

    # 使用 KFold 生成五个不重叠的分组
    fold_indices = [test_idx for _, test_idx in kfold.split(features.cpu())] 

    return city_trans_dataset, fold_indices

