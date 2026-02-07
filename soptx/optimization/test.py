import matplotlib.font_manager
from matplotlib.font_manager import FontProperties

def list_useful_fonts():
    print("=" * 60)
    print("正在扫描系统中的中文字体和常用英文字体...")
    print("=" * 60)
    
    # 获取所有系统字体
    fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    
    chinese_fonts = []
    serif_fonts = []
    
    for font_path in fonts:
        try:
            prop = matplotlib.font_manager.FontProperties(fname=font_path)
            name = prop.get_name()
            
            # 简单的关键词过滤
            lower_name = name.lower()
            
            # 检查中文常见关键词
            if any(x in lower_name for x in ['hei', 'song', 'kai', 'ming', 'noto sans cjk', 'wenquanyi', 'droid sans fallback']):
                chinese_fonts.append((name, font_path))
            
            # 检查 Serif/Times
            if 'times' in lower_name or 'liberation serif' in lower_name:
                serif_fonts.append((name, font_path))
                
        except:
            continue

    print("【发现的中文字体候选】：")
    if not chinese_fonts:
        print("  (未找到显式的中文字体，请务必手动上传 SimHei.ttf)")
    for name, path in sorted(chinese_fonts):
        print(f"  名称: {name:<25} | 路径: {path}")

    print("\n【发现的 Times/Serif 字体候选】：")
    if not serif_fonts:
        print("  (未找到 Times New Roman，请务必手动上传 times.ttf)")
    for name, path in sorted(serif_fonts):
        print(f"  名称: {name:<25} | 路径: {path}")
        
    print("=" * 60)

if __name__ == "__main__":
    list_useful_fonts()