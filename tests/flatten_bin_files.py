import os
import sys
import shutil
from pathlib import Path

def flatten_files(input_dir: str, target_dir: str):
    input_path = Path(input_dir).resolve()
    target_path = Path(target_dir).resolve()

    target_path.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(input_path):
        for filename in files:
            src_file = Path(root) / filename

            # 解析软链接 → 复制真实文件
            if src_file.is_symlink():
                real_file = src_file.resolve()
                if not real_file.is_file():
                    continue
                src_file = real_file

            dst_file = target_path / src_file.name

            try:
                shutil.copy2(src_file, dst_file)
                print(f"copy: {src_file} --> {dst_file}")
            except Exception as e:
                print(f"skip {src_file}: {str(e)}")

    print("\nFlattening complete! All files have been copied to:", target_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage:")
        print("  python flatten_bin_files.py input_dir target_dir")
        print("example:")
        print("  python flatten_bin_files.py build/install build/final")
        sys.exit(1)

    input_dir = sys.argv[1]
    target_dir = sys.argv[2]
    flatten_files(input_dir, target_dir)