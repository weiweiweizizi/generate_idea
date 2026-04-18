"""
运行所有实验并生成汇总表格

执行12个配置 (2矩阵类型 × 2方向 × 3任务) 并生成对比表
"""

import sys
sys.path.insert(0, '/home/weizilin/generate_idea/scripts/val_codebook')

import subprocess
from pathlib import Path
import json
import pandas as pd


def run_all_experiments():
    """运行所有实验"""
    print("=" * 60)
    print("运行所有验证实验")
    print("=" * 60)

    experiments = [
        ('exp1_dataset_classification', 'run.py'),
        ('exp2_side_classification', 'run.py'),
        ('exp3_severity_classification', 'run.py'),
    ]

    for exp_name, script in experiments:
        print(f"\n{'='*50}")
        print(f"Running: {exp_name}")
        print(f"{'='*50}")

        script_path = Path(__file__).parent / exp_name / script
        result = subprocess.run(['python', str(script_path)],
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {exp_name}:")
            print(result.stderr)
        else:
            print(result.stdout)


def collect_results():
    """收集所有实验结果"""
    base_dir = Path(__file__).parent

    all_results = []

    for exp_name in ['exp1_dataset_classification', 'exp2_side_classification', 'exp3_severity_classification']:
        exp_dir = base_dir / exp_name / 'output'

        for config_file in exp_dir.glob('results_*.json'):
            with open(config_file) as f:
                results = json.load(f)

            all_results.append({
                'experiment': exp_name,
                'config': config_file.stem.replace('results_', ''),
                'accuracy': results.get('accuracy_mean'),
                'accuracy_std': results.get('accuracy_std'),
                'f1': results.get('f1_mean'),
                'f1_std': results.get('f1_std'),
                'auc': results.get('auc_mean'),
                'auc_std': results.get('auc_std'),
            })

    return pd.DataFrame(all_results)


def generate_summary_table():
    """生成汇总对比表"""
    print("\n" + "=" * 80)
    print("实验结果汇总对比表")
    print("=" * 80)

    df = collect_results()

    if df.empty:
        print("No results found. Run experiments first.")
        return

    # 按实验分组打印
    for exp_name in ['exp1_dataset_classification', 'exp2_side_classification', 'exp3_severity_classification']:
        print(f"\n{exp_name}:")
        print("-" * 80)

        exp_df = df[df['experiment'] == exp_name].copy()

        # 打印表头
        print(f"{'Config':<20} {'Acc':<14} {'F1':<14} {'AUC':<14}")
        print("-" * 80)

        for _, row in exp_df.iterrows():
            config = row['config']
            acc = f"{row['accuracy']:.3f}±{row['accuracy_std']:.3f}" if row['accuracy'] else "N/A"
            f1 = f"{row['f1']:.3f}±{row['f1_std']:.3f}" if row['f1'] else "N/A"
            auc = f"{row['auc']:.3f}±{row['auc_std']:.3f}" if row['auc'] else "N/A"
            print(f"{config:<20} {acc:<14} {f1:<14} {auc:<14}")

    # 保存到CSV
    output_path = Path(__file__).parent / 'summary_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\n汇总已保存: {output_path}")

    # 打印完整的对比矩阵
    print("\n" + "=" * 80)
    print("完整对比 (Acc)")
    print("=" * 80)

    pivot = df.pivot_table(index='config', columns='experiment', values='accuracy')
    print(pivot.to_string())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Run all experiments')
    parser.add_argument('--summary', action='store_true', help='Generate summary table')
    args = parser.parse_args()

    if args.run:
        run_all_experiments()
    if args.summary or not args.run:
        generate_summary_table()
