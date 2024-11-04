import os
import json
import numpy as np
from tqdm import tqdm
# 文件夹路径
json_folder = r'F:\Dataset\3DRealCAR\sample_data\car_01\camera_paras'
images_folder = r'F:\Dataset\3DRealCAR\sample_data\car_01\images'

# 创建一个集合来保存images文件夹中的frame_id
frame_ids = set()
for filename in os.listdir(images_folder):
    if filename.startswith("frame_") and filename.endswith(".jpg"):
        frame_id = filename.split('_')[1].split('.')[0]  # 提取frame id
        frame_ids.add(frame_id)

# 转换为排序列表，确保frame_id是按升序排列
sorted_frame_ids = sorted(frame_ids)

# 用于保存内参和外参的列表
intrinsics_matrices = []
extrinsics_matrices = []

# 遍历json文件夹中的frame编号.json文件
for frame_id in tqdm(sorted_frame_ids, desc="Processing JSON files"):
    json_filename = f"frame_{frame_id}.json"
    json_path = os.path.join(json_folder, json_filename)
    
    # 检查对应的json文件是否存在
    if os.path.exists(json_path):
        # 读取json文件
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        
        # 获取内参矩阵 (位于'intrinsics'字段)
        intrinsics = np.array(data.get('intrinsics', [])).reshape(3, 3)
        
        # 获取外参矩阵 (位于'cameraPoseARFrame'字段)
        extrinsics = np.array(data.get('cameraPoseARFrame', [])).reshape(4, 4)
        
        # 将读取到的内参和外参矩阵添加到对应的列表中
        intrinsics_matrices.append(intrinsics)
        extrinsics_matrices.append(extrinsics)

# 转换为numpy数组
intrinsics_matrices = np.array(intrinsics_matrices)
extrinsics_matrices = np.array(extrinsics_matrices)

# 将数据保存到与上述结构相同的npy文件中，内参和外参矩阵都属于FRONT
intrinsics_dict = {'FRONT': intrinsics_matrices}
extrinsics_dict = {'FRONT': extrinsics_matrices}

# 保存为npy文件
np.save(r'F:\Dataset\3DRealCAR\sample_data\car_01\intrinsics.npy', intrinsics_dict)
np.save(r'F:\Dataset\3DRealCAR\sample_data\car_01\extrinsics.npy', extrinsics_dict)

print("Intrinsics and extrinsics matrices have been saved successfully.")
