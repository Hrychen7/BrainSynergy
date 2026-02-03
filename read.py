import os
import re
import pandas as pd

def collect_results(base_root='./results'):
    all_data = []
    # 遍历任务层 (NC_MCI, MCI_AD)
    if not os.path.exists(base_root):
        print(f"错误: 找不到目录 {base_root}")
        return

    for task in os.listdir(base_root):
        task_path = os.path.join(base_root, task)
        if not os.path.isdir(task_path): continue

        # 遍历模型层 (TAN, MGCN, CrossGNN)
        for model in os.listdir(task_path):
            model_path = os.path.join(task_path, model)
            summary_file = os.path.join(model_path, 'final_summary.txt')
            
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    content = f.read()
                    
                # 使用正则提取各项指标的 Mean 和 Std
                # 匹配格式: Metric | Mean | Std
                metrics = {}
                patterns = {
                    'ACC': r'Accuracy\s+\|\s+([\d\.]+)%?\s+\|\s+([\d\.]+)%?',
                    'SEN': r'Sensitivity\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)',
                    'SPE': r'Specificity\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)',
                    'F1':  r'F1-Score\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)',
                    'AUC': r'AUC\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)'
                }
                
                row = {'Task': task, 'Model': model}
                for name, pat in patterns.items():
                    match = re.search(pat, content)
                    if match:
                        mean, std = match.groups()
                        row[name] = f"{mean} ± {std}"
                
                all_data.append(row)

    if not all_data:
        print("未发现有效的 final_summary.txt 文件。")
        return

    # 转为 DataFrame 并美化输出
    df = pd.DataFrame(all_data)
    # 排序让对比更清晰
    df = df.sort_values(by=['Task', 'Model'])
    
    print("\n" + "="*80)
    print("模型性能全对比汇总 (Mean ± Std)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    # 保存为 CSV 方便导入 Excel 绘图
    df.to_csv('all_models_comparison.csv', index=False)
    print("\n汇总表格已保存至: all_models_comparison.csv")

if __name__ == "__main__":
    collect_results()