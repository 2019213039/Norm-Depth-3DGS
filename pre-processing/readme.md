# 1. Original Data
## 1.sparse\0
'''
EVERY CAR'S ORIGINAL DATA PATH\colmap_processed\pcd_rescale\sparse\0
'''
This is the rescaled colmap data, which has cameras.bin, images.bin and points3D.bin
You can directly copy this sparse\0 into your own datafile
## 2.Masks data
'''
EVERY CAR'S ORIGINAL DATA PATH\colmap_processed\masks\sam
'''
You can use the binary mask in the file path for training, but you need to change the name of it into 000000.png.png ......
## 3.Depth data
'''
EVERY CAR'S ORIGINAL DATA PATH\depth_<frame_id>.png
'''
### 1.Save the original PNG type depth data into .NPY
### 2.Rescale the unit from mm to meter. 
'''
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
                npy_output_path = depth_image_path.replace('.png', '.npy')
                np.save(npy_output_path, depth_array_in_meters.astype(np.float32))
                print(f"Saved {filename} as float32 .npy file.")
'''
### 3.Rename the depth data into <frame_id>.png.npy
## 4.Normal data
use depth2norm.py to convert depth map into normal map
Before you doing this, you need to get the 'intrinsics.npy' from camera_poses.py. But you can also extract the intrinsics matrix directly.

