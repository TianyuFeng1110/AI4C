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

# 交通枢纽区 骑行：共享单车周未平均路段骑行量，光：年平均夜间灯光dnb辐射值
# ['Land01', 'BuiDen', 'FAR', 'BuiHight', 'BusStop']

model = joblib.load("../model/xgboost/model_light_dnb__bk_str_we.pkl")

class MyProblem(Problem):

    def __init__(self):
        super().__init__(
            n_var=26,      # 决策变量维度
            n_obj=2,      # 目标数量
            n_constr=0,   # 约束数量
            # 住宅用地百分比, 建筑总底面积占栅格面积的比例,建筑总容积 / 栅格总面积(1km * 1km）,平均建筑高度, 每个栅格中公交站数量
            # xl=np.array([0, 50, 4.0, 100, 50]),  # 变量下界
            # xu=np.array([5, 70, 8.0, 300, 100])  # 变量上界
            xl=np.array([0.20 ,0.30 ,0.10 ,0.08 ,0.05 ,0.5 ,0.60 ,0.2 ,50 , 0 ,120 ,60  ,8  ,80  ,60  ,8  ,4 ,2 ,0.40 ,3.0 ,40 ,0.8  ,4.0 ,10 ,4  ,7]),  # 变量下界
            xu=np.array([0.35 ,0.50 ,0.20, 0.20 ,0.10 ,0.7 ,0.80 ,0.4 ,100, 9 ,200 ,120 ,20 ,150 ,120 ,12 ,8 ,4 ,0.50 ,5.0 ,80 ,1.25 ,6.7 ,20 ,12 ,15])  # 变量上界
        )

        self.model_path = "../model/dl/saved_models/model_fold0.pt"
        self.data_file = "../model/dl/shanghai_data_v3.xlsx"
        self.device = utils.select_device(3)
        self.city_trans_dataset = utils.get_five_fold_datasets(self.data_file)
        # 展示区域对应id号 如商务区展示5011
        self.id = 5115
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
                fe = utils.get_all_data_from_X(torch.tensor(X[n]).float(), id_idx, self.features) # 新的features，维度一致
                fe = torch.tensor(fe).to(self.device)
                # 模型输出处理
                outputs = self.model(fe.float(), self.coordinates)         # 所有的坐标都是与ids对应的坐标一致
                light_pred = outputs['light'].detach().cpu().numpy()
                temp_pred = outputs['temp'].detach().cpu().numpy()
                walk_pred = outputs['walk'].detach().cpu().numpy()
                # lst_day_c temp_pred[0] ,  bk_st_we walk_pred[1]
                light = light_pred[:, 0] # 年平均夜间灯光dnb辐射值
                ride = walk_pred[:, 4] # 共享单车周未平均路段骑行量

                arrays.append(np.column_stack([light, ride])[id_idx])

        light = np.vstack(arrays)[:, 0]  # 年平均夜间灯光dnb辐射值
        ride = np.vstack(arrays)[:, 1]  # 共享单车周未平均路段骑行量

        # 条件1：共享单车周未平均路段骑行量
        QRmax, QRmin = 250, 120
        conditions = [ride < QRmin, (ride >= QRmin) & (ride <= QRmax), ride > QRmax]
        choices = [((ride - QRmin) ** 2), 0, ((ride - QRmax) ** 2)]  # 不同条件对应不同值
        f1 = np.select(conditions, choices)

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
np.savetxt("./Transport.txt", res.X, fmt="%.6f", delimiter=" ")