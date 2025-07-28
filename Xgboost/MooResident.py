import sys

import numpy as np
import joblib
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions  # 注意这里是ref_dirs不是reference_direction
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from setuptools.command.easy_install import sys_executable

# 住宅区 热：夜间地表温度，骑行： 共享单车工作日平均路段骑行量
# ['Land01', 'BuiDen', 'FAR', 'BuiHight', 'BusStop']

model = joblib.load("../model/xgboost/model_lst_night__bk_str_wk.pkl")

class MyProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=5,      # 决策变量维度
            n_obj=2,      # 目标数量
            n_constr=0,   # 约束数量
            # 住宅用地百分比, 建筑总底面积占栅格面积的比例,建筑总容积 / 栅格总面积(1km * 1km）,平均建筑高度, 每个栅格中公交站数量
            xl=np.array([0.50, 0.25, 1.0, 24, 2.5]),  # 变量下界
            xu=np.array([0.60, 0.35, 2.0, 54, 3.3])  # 变量上界
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F_pred = model.predict(X)
        temp = F_pred[:, 0] # 夜间地表温度
        ride = F_pred[:, 1] # 共享单车工作日平均路段骑行量

        # 条件1：夜间地表温度
        QTmax, QTmin = 26, 20  # 夜间最适温度区间为20-26
        a, b = 2, 1 # ab分别为温度惩罚系数，当温度大于适宜温度区间时过热的惩罚要大于温度小于适宜温度区间过冷的惩罚
        conditions = [temp < QTmin, (temp >= QTmin) & (temp <= QTmax), temp > QTmax]
        choices = [b * (temp - QTmin)**2, 0, a * (temp - QTmax)**2]  # 不同条件对应不同值
        f1 = np.select(conditions, choices)

        # 条件2：共享单车工作日平均路段骑行量
        QRmax, QRmin = 120, 60
        conditions = [ride < QRmin, (ride >= QRmin) & (ride <= QRmax), ride > QRmax]
        choices = [(ride - QRmin) ** 2, 0, (ride - QRmax) ** 2]  # 不同条件对应不同值
        f2 = np.select(conditions, choices)

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
    ('n_gen', 100),
    seed=1,
    verbose=True
)

print("最优解的目标值：")
print(res.F)
print("对应的决策变量：")
print(res.X)