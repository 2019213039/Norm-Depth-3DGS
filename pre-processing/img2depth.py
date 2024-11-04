import os
import numpy as np
from PIL import Image

def filter_and_delete_unmatched_depths(images_folder, depths_folder):
    """
    筛选并删除不存在于 images 文件夹中的 frame id 的 depth 图像。
    
    参数:
    images_folder (str): 包含 image 文件的文件夹路径
    depths_folder (str): 包含 depth 文件的文件夹路径
    """
    # 获取 images 文件夹中的所有文件名，并提取 frame id
    frame_ids = set()
    for filename in os.listdir(images_folder):
        if filename.startswith("frame_") and filename.endswith(".jpg"):
            frame_id = filename.split('_')[1].split('.')[0]  # 提取frame id
            frame_ids.add(frame_id)

    # 遍历 depths 文件夹中的文件，删除不存在于 frame ids 的图片
    for filename in os.listdir(depths_folder):
        if filename.startswith("depth_") and filename.endswith(".png"):
            depth_id = filename.split('_')[1].split('.')[0]
            if depth_id not in frame_ids:
                # 删除在 frame ids 中不存在的文件
                os.remove(os.path.join(depths_folder, filename))
                print(f"Deleted {filename}")

    print("筛选和删除完成！")


def upsample_depth_images(depths_folder, target_size=(1920, 1440)):
    """
    使用双线性插值将深度图像上采样到指定大小。
    
    参数:
    depths_folder (str): 包含 depth 图像的文件夹路径
    target_size (tuple): 上采样后的目标大小
    """
    for filename in os.listdir(depths_folder):
        if filename.endswith(".png"):
            # 打开 depth 图片
            depth_image_path = os.path.join(depths_folder, filename)
            with Image.open(depth_image_path) as depth_image:
                # 检查图像大小是否为 256x192
                if depth_image.size == (256, 192):
                    # 使用双线性插值进行上采样
                    upsampled_image = depth_image.resize(target_size, Image.BILINEAR)
                    
                    # 保存上采样后的图片，覆盖原图
                    upsampled_image.save(depth_image_path)
                    print(f"Upsampled and saved {filename}")
                else:
                    print(f"Skipping {filename}, size is not 256x192")
    
    print("图片上采样处理完成！")


def convert_depth_to_float32(folder_path):
    """
    将深度图像的像素值从毫米转换为米，并保存为 float32 格式的 .npy 和 .tiff 文件。
    
    参数:
    folder_path (str): 包含 depth 图像的文件夹路径
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            # 构建深度图片的完整路径
            depth_image_path = os.path.join(folder_path, filename)
            
            # 打开深度图片并转换为 numpy 数组
            with Image.open(depth_image_path) as depth_image:
                depth_array = np.array(depth_image)
                
                # 将深度值从毫米转换为米
                depth_array_in_meters = depth_array / 1000.0
                
                # 保存为 .npy 格式
                npy_output_path = depth_image_path.replace('.png', '_meters.npy')
                np.save(npy_output_path, depth_array_in_meters.astype(np.float32))
                print(f"Saved {filename} as float32 .npy file.")
                
                # 保存为 .tiff 格式
                tiff_output_path = depth_image_path.replace('.png', '_meters.tiff')
                Image.fromarray(depth_array_in_meters.astype(np.float32)).save(tiff_output_path)
                print(f"Saved {filename} as float32 .tiff file.")
    
    print("深度图转换和保存为 float32 格式完成！")


def process_depth_images(images_folder, depths_folder):
    """
    依次进行删除不匹配的深度图，上采样，转换为 float32 并保存。
    
    参数:
    images_folder (str): 包含 images 的文件夹路径
    depths_folder (str): 包含 depth 图像的文件夹路径
    """
    # 第一步：筛选并删除不存在的深度图像
    filter_and_delete_unmatched_depths(images_folder, depths_folder)
    
    # 第二步：对深度图像进行上采样
    upsample_depth_images(depths_folder, target_size=(1920, 1440))
    
    # 第三步：将深度图转换为 float32 并保存为 .npy 和 .tiff 格式
    convert_depth_to_float32(depths_folder)

# 示例调用
images_folder = r'F:\Dataset\3DRealCAR\sample_data\car_01\images'
depths_folder = r'F:\Dataset\3DRealCAR\sample_data\car_01\depths'

process_depth_images(images_folder, depths_folder)

