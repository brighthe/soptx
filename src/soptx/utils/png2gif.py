from pathlib import Path
from typing import Optional
from PIL import Image

def png_to_gif(folder_name: str, output_name: str = 'output.gif', duration: int = 100) -> Optional[Path]:
    """
    将指定文件夹内的所有 PNG 图片转换为 GIF 动画
    
    Parameters:
    -----------
    folder_name : str
        文件夹名称
    output_name : str
        输出 GIF 文件名，默认为 'output.gif'
    duration : int
        每帧显示时间 (毫秒)，默认为 100
        
    Returns:
    --------
    Optional[Path]
        成功时返回生成的 GIF 文件路径，失败时返回 None
    """
    # 获取基础路径
    current_file = Path(__file__)
    base_dir = current_file.parent.parent / 'vtu' / folder_name
    
    # 获取该文件夹下所有的 png 文件
    png_files = sorted(base_dir.glob('*.png'))
    
    if not png_files:
        print(f"在 {base_dir} 中没有找到任何 PNG 文件")
        return None
    
    print(f"找到 {len(png_files)} 个 PNG 文件")
    
    # 打开所有图片
    images = [Image.open(str(png_file)) for png_file in png_files]
    
    # 确保输出文件名以 .gif 结尾
    if not output_name.endswith('.gif'):
        output_name += '.gif'
    
    # 输出 GIF 文件路径
    output_path = base_dir / output_name
    
    # 创建 GIF
    images[0].save(
        str(output_path),
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    
    print(f"GIF 动图已生成: {output_path}")
    return output_path


# 使用示例
if __name__ == "__main__":
    # 转换 test 文件夹中的所有 PNG 为 GIF
    # png_to_gif('canti_6_17')
    
    # 指定输出文件名
    # png_to_gif('canti_6_17', 'density_animation.gif')
    
    # 或者指定不同的文件夹、文件名和播放速度
    png_to_gif('mbb2d_6_19', 'density_animation.gif', duration=100)