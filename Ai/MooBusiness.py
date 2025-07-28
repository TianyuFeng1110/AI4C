import sys

import numpy as np
import joblib
import torch
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions  # 注意这里是ref_dirs不是reference_direction
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent  # 关键点：两个.parent
sys.path.append(str(project_root))

import utils

# 中央商务区 热：日间地表温度，光：年平均夜间灯光dnb辐射值
# ['Land01', 'Landmix', 'BuiDen', 'FAR', 'BuiHight', 'BusStop']

model = joblib.load("../model/xgboost/model_lst_day_c_light_dnb_.pkl")

class MyProblem(Problem):

    def __init__(self):
        super().__init__(
            n_var=26,      # 决策变量维度
            n_obj=2,      # 目标数量
            n_constr=0,   # 约束数量
            # 住宅用地百分比, 建筑总底面积占栅格面积的比例,建筑总容积 / 栅格总面积(1km * 1km）,平均建筑高度, 每个栅格中公交站数量
            # xl=np.array([0, 50, 4.0, 100, 50]),  # 变量下界
            # xu=np.array([5, 70, 8.0, 300, 100])  # 变量上界
            xl=np.array([0    ,0.5 ,0   , 0.06 ,0.05 ,0.6 ,0.70 ,0.2 ,80 , 0 ,250 ,80  ,12 ,150 ,80  ,10 ,5  ,2 ,0.60 ,5.0 ,80  ,0.8  ,3.3 ,1.12 ,1.7 ,3.4]),  # 变量下界
            xu=np.array([0.05 ,0.7 ,0.05, 0.15 ,0.15 ,0.8 ,0.85 ,0.4 ,150, 9 ,400 ,150 ,25 ,300 ,150 ,20 ,10 ,5 ,0.70 ,8.0 ,120 ,1.25 ,5.0 ,1.67 ,2.5 ,5.0])  # 变量上界
        )

        self.model_utils = utils
        self.model_path = "../model/dl/saved_models/model_fold0.pt"
        self.data_file = "../model/dl/shanghai_data_v3.xlsx"
        self.device = self.model_utils.select_device(0)
        self.city_trans_dataset = self.model_utils.get_five_fold_datasets(self.data_file)
        # 展示区域对应id号 如商务区展示5011
        self.id = 5011
        #获取数据
        self.features = self.city_trans_dataset.features.to(self.device)
        self.coordinates = self.city_trans_dataset.coordinates.to(self.device)
        self.ids = self.city_trans_dataset.ids.cpu().numpy()
        # 模型加载
        self.model = torch.load(self.model_path, map_location=self.device, weights_only=False).to(self.device)
        self.model.eval()

    def _evaluate(self, X, out, *args, **kwargs): # 101,26
        with torch.no_grad():
            id_idx = np.where(self.ids == self.id)[0][0] # 从数据的id找到对应的索引和坐标，如商务区5011对应的id是多少
            arrays = []
            # 数据拼接
            for n in range(X.shape[0]):
                # print("数据拼接")
                fe = utils.get_all_data_from_X(torch.tensor(X[n]).float(), id_idx, self.features) # 新的features，维度一致
                fe = torch.tensor(fe).to(self.device)
                # 模型输出处理
                outputs = self.model(fe.float(), self.coordinates)         # 所有的坐标都是与ids对应的坐标一致
                light_pred = outputs['light'].detach().cpu().numpy()
                temp_pred = outputs['temp'].detach().cpu().numpy()
                # walk_pred = outputs['walk'].detach().cpu().numpy()
                # lst_day_c temp_pred[0] ,  bk_st_we walk_pred[1]
                temp = temp_pred[:, 0] # 日间地表温度
                light = light_pred[:, 0] # 年平均夜间灯光dnb辐射值
                # print("e")

                arrays.append(np.column_stack([temp, light])[id_idx])

        temp = np.vstack(arrays)[:, 0] # 日间地表温度
        light = np.vstack(arrays)[:, 1] # 年平均夜间灯光dnb辐射值

        # 条件1：日间地表温度最优 f(Q)=∣Q−Q max∣+∣Q−Q min∣ Qmax，Qmin分别为12-26℃
        f1 = np.abs(temp - 26) + np.abs(temp - 12)
        # 条件2：年平均夜间灯光dnb辐射值 f(L) = (L - 65)**2
        f2 = (light - 65) ** 2

        out["F"] = np.column_stack([f1, f2])

# 使用自定义问题
problem = MyProblem()

# 2. 生成参考方向（关键修正点）
# get_reference_directions(生成方法, 目标数, 分割数)
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=100)

# 3. 配置NSGA-III算法
algorithm = NSGA3(
    ref_dirs=ref_dirs,
    pop_size=101, # pop_size表示每一代（Generation）中维持的候选解（个体）的数量
)

# 4. 运行优化
res = minimize(
    problem,
    algorithm,
    ('n_gen', 300),
    seed=1,
    verbose=True
)

print("最优解的目标值：")
print(res.F)
print("对应的决策变量：")
print(res.X)

# 保存到文本文件（默认空格分隔，保留4位小数）
np.savetxt("./Business.txt", res.X, fmt="%.6f", delimiter=" ")