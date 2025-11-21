import os
import json

def export_folder_names_to_json(root_path, output_file="./folders.json"):
    """
    提取指定路径下所有子文件夹的名称，并导出为 JSON 文件。
    :param root_path: 要扫描的文件夹路径
    :param output_file: 输出 JSON 文件名
    """
    # 检查路径是否存在
    if not os.path.exists(root_path):
        print(f"路径不存在: {root_path}")
        return

    # 获取所有子文件夹名称
    folder_names = []
    for item in os.listdir(root_path):
        full_path = os.path.join(root_path, item)
        if os.path.isdir(full_path):
            folder_names.append(item)

    # 导出为 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(folder_names, f, ensure_ascii=False, indent=4)

    print(f"✅ 成功提取 {len(folder_names)} 个文件夹名称，已保存至 {output_file}")

if __name__ == "__main__":
    # 修改这里为你要扫描的目录
    # target_path = r"/home/yilin/yilin-DPO/output/llava-v1.5-7b"  # Windows 示例
    target_path = r"/home/yilin/yilin-DPO/output/llava-v1.6-vicuna-7b"  # Windows 示例
    # target_path = "/home/username/path" # Linux 示例
    export_folder_names_to_json(target_path)
