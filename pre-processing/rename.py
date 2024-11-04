'''
1. F:\Dataset\3DRealCAR\sample_data\car_01\images里的文件重命名，将原本的frame_XXXXX改为位的frame id。比如，之前的frame_00000.jpg改为000000.jpg;之后将文件夹内的所有命名按照0到文件个数-1的方式重新命名，000000.jpg,000001.jpg...
2. F:\Dataset\3DRealCAR\sample_data\car_01\depths里的png文件重命名，将原本depth_00000.png改为000000.png。之后将所有frame id按照从小到达重新排列，并以0开始，顺序加1（比如原来的文件是00000.00006，00008改完后就是000000，000001，000002）
3. F:\Dataset\3DRealCAR\sample_data\car_01\depths里的npy文件重命名位重新排序后的frame id.png.npy
4. F:\Dataset\3DRealCAR\sample_data\car_01\depths里的tiff文件重命名为重新排序后的frame id.tiff
5. F:\Dataset\3DRealCAR\sample_data\car_01\norms里的文件重命名位重新排序后的frame id.png.npy
6. F:\Dataset\3DRealCAR\sample_data\car_01\mask里的所有文件，比如frame_00000.jpg重命名为000000.png.png并且所有frame重新排序
'''
import os
from tqdm import tqdm

def rename_files_in_directory(directory, prefix, extension, new_extension=None):
    files = sorted([f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)])
    for i, file in enumerate(tqdm(files, desc=f"Renaming {prefix} files in {directory}")):
        new_filename = f"{i:06d}{new_extension or extension}"
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)

def rename_npy_files(directory):
    npy_files = sorted([f for f in os.listdir(directory) if f.endswith('.npy') and not f.endswith('.png.npy')])
    for i, file in enumerate(tqdm(npy_files, desc=f"Renaming .npy files in {directory}")):
        new_filename = f"{i:06d}.png.npy"
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)

def rename_tiff_files(directory):
    tiff_files = sorted([f for f in os.listdir(directory) if f.endswith('.tiff')])
    for i, file in enumerate(tqdm(tiff_files, desc=f"Renaming .tiff files in {directory}")):
        new_filename = f"{i:06d}.tiff"
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)

def modify_colmap_txt(file_path, output_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(output_path, 'w') as f:
        image_id = 0
        for idx, line in enumerate(tqdm(lines, desc="Processing COLMAP .txt file")):
            if line.startswith("#"):  # Write comment lines unchanged
                f.write(line)
            elif idx % 2 == 0:  # Only modify odd-numbered lines (0, 2, 4,... zero-indexed)
                parts = line.split()
                if len(parts) >= 10:  # Make sure it's an image-related line
                    parts[-1] = f"{image_id:06d}.jpg"  # Modify the file name
                    image_id += 1
                f.write(' '.join(parts) + '\n')
            else:
                f.write(line)  # Even-numbered lines (1, 3, 5,...) are written unchanged

def main():
    # base_dir = r"F:\Dataset\3DRealCAR\sample_data\car_01"
    
    # # Step 1: Rename images (jpg files)
    # rename_files_in_directory(os.path.join(base_dir, "images"), "frame_", ".jpg")
    
    # # Step 2: Rename depths (png files)
    # rename_files_in_directory(os.path.join(base_dir, "depths"), "depth_", ".png")
    
    # # Step 3: Rename depths (npy files)
    # rename_npy_files(os.path.join(base_dir, "depths"))
    
    # # Step 4: Rename depths (tiff files)
    # rename_tiff_files(os.path.join(base_dir, "depths"))
    
    # # Step 5: Rename norms (npy files)
    # rename_npy_files(os.path.join(base_dir, "norms"))
    
    # # Step 6: Rename masks (jpg files) and change extension to `.png.png`
    # rename_files_in_directory(os.path.join(base_dir, "mask"), "frame_", ".jpg", new_extension=".png.png")

    # Step 7: Modify the COLMAP txt file with new names
    colmap_txt_path = r"F:\Dataset\3DRealCAR\sample_data\car_01\sparse\0\images.txt"
    modified_txt_path = r"F:\Dataset\3DRealCAR\sample_data\car_01\sparse\0\images_new.txt"
    modify_colmap_txt(colmap_txt_path, modified_txt_path)

if __name__ == "__main__":
    main()
