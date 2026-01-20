from PIL import Image
from pathlib import Path

img_folder_path = Path.home() / "hlthesis/paper/SOPTX_cicp_clean_manuscript/figures"

print(f"Target Directory: {img_folder_path}")

# --- 转换逻辑 ---
if not img_folder_path.exists():
    print("Error: 找不到文件夹！请检查路径是否正确。")
    exit()

# 遍历所有 PNG 文件
files = list(img_folder_path.glob("*.png"))
if not files:
    print("Warning: 该目录下没有找到 .png 文件。")

for img_path in files:
    try:
        # 打开图片
        with Image.open(img_path) as img:
            # 构造输出文件名 (.png -> .eps)
            eps_path = img_path.with_suffix('.eps')
            
            # 转换模式：EPS 不支持透明 (RGBA)，需转为 RGB
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                # 处理透明背景，将其变白
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert("RGB")
            
            # 保存为 EPS
            img.save(eps_path, "EPS", quality=100)
            print(f"[OK] Converted: {img_path.name} -> {eps_path.name}")
            
    except Exception as e:
        print(f"[Error] Failed to convert {img_path.name}: {e}")

print("All done!")