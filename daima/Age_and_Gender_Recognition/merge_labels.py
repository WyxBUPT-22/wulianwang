import pandas as pd

def merge_label_files(label_files, output_file):
    """
    合并多个标签文件到一个数据框，并保存为新文件。
    :param label_files: 标签文件路径列表
    :param output_file: 合并后的输出文件路径
    """
    dataframes = []
    for file in label_files:
        df = pd.read_csv(file, sep="\t")  # 读取标签文件
        dataframes.append(df)

    merged_data = pd.concat(dataframes, ignore_index=True)  # 合并所有文件
    merged_data.to_csv(output_file, sep="\t", index=False)  # 保存为新文件
    print(f"合并完成，保存为 {output_file}")


if __name__ == "__main__":
    # 标签文件列表
    label_files = [
        "fold_0_data.txt",
        "fold_1_data.txt",
        "fold_2_data.txt",
        "fold_3_data.txt",
        "fold_4_data.txt"
    ]

    # 合并文件输出路径
    output_file = "all_data.txt"

    # 合并标签文件
    merge_label_files(label_files, output_file)
