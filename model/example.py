import torch
import numpy as np
from cv_dataset import get_five_fold_datasets


# 训练设备(使用GPU)
def select_device(gpu_id=None):
    if gpu_id is None:
        print("select device CPU")
        return torch.device("cpu")
    if torch.cuda.is_available():
        print("select device GPU")
        return torch.device("cuda:" + str(gpu_id))
    print("have to select device CPU")
    return torch.device("cpu")


def predict_and_display_results(model_path, data_file, device="cpu", top_n=5):
    """
    加载模型并进行预测，展示前top_n个样本的预测结果、ID和坐标
    """
    # 1. 加载模型
    my_model = torch.load(model_path, map_location=device, weights_only=False)
    my_model = my_model.to(device)
    print(my_model)
    my_model.eval()

    # 2. 加载数据集（复用您的数据生成函数）
    city_trans_dataset, _ = get_five_fold_datasets(data_file)  # fold_indices未使用

    # 3. 获取整个数据集的预测结果
    with torch.no_grad():
        features = city_trans_dataset.features.to(device)
        coordinates = city_trans_dataset.coordinates.to(device)
        ids = city_trans_dataset.ids.cpu().numpy()
        outputs = my_model(features, coordinates)

        # 将预测结果移回CPU并转为numpy
        light_pred = outputs['light'].cpu().numpy()
        temp_pred = outputs['temp'].cpu().numpy()
        walk_pred = outputs['walk'].cpu().numpy()

    # 4. 展示前top_n个样本
    print(f"\n展示前{top_n}个样本的预测结果：")
    for i in range(min(top_n, len(ids))):
        print(f"样本{i + 1}:")
        print(f"  ID: {ids[i]}")
        print(f"  坐标: ({coordinates[i][0].item():.2f}, {coordinates[i][1].item():.2f})")
        print(f"  光污染预测: {light_pred[i][0]:.4f}")
        print(f"  温度预测: [{temp_pred[i][0]:.4f}, {temp_pred[i][1]:.4f}]")
        print(f"  慢行环境预测: {walk_pred[i].tolist()}")
        print("-" * 50)


# 使用示例
if __name__ == "__main__":
    model_path = "/home/jaily/jaily/city_grid_multi_prediction_transformer/cross_vaildation/saved_models/model_fold0.pt"
    data_file = "shanghai_data_v3.xlsx"
    device = select_device(2)
    predict_and_display_results(model_path, data_file, device=device, top_n=5)