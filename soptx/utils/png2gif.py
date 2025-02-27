from PIL import Image
"""
This script converts a series of PNG images into a single GIF animation.
"""

# 图片文件路径列表
image_files = []
# 图片文件的数量
files_num = 105
# 图片文件的基础目录
base_dir = '/home/heliang/FEALPy_Development/soptx/soptx/vtu/gif'
# 生成图片文件路径列表
for i in range(files_num):
    image_files.append(f'{base_dir}/2d_' + str(i) + '.png')

# 打开图片并存入一个列表
images = [Image.open(image_file) for image_file in image_files]

# 每张图片的显示时间（单位是 ms）
duration = 100 

# 创建 GIF 动图
images[0].save(f'{base_dir}/output.gif', save_all=True, append_images=images[1:], duration=duration, loop=0)

print("GIF 动图已生成：output.gif")