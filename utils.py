# 数据处理和可视化
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 统计检验
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, pearsonr, spearmanr, norm
from statsmodels.stats import proportion, contingency_tables
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.contingency_tables import mcnemar

# 机器学习
from sklearn import metrics, feature_extraction
from sklearn.metrics import cohen_kappa_score, jaccard_score
from sklearn.feature_extraction.text import CountVectorizer

# 其他工具
import re
import os
import json
import openpyxl
from glob import glob
from tqdm import tqdm
from itertools import product, combinations
from collections import defaultdict

# 取消warnings
import warnings
warnings.filterwarnings("ignore")

# 参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_version', type=str, default='mmlu_pro', choices=["mmlu_pro", "winogrande", "hellaswag", "GaoKao", "BBH"])
parser.add_argument('--draw_corr_fig', type=int, default=0, help='Whether to draw correlation figures between models.')
parser.add_argument('--force_read', type=int, default=1, help='Whether to force read data instead of using cache.')
args = parser.parse_args()


def task_name_func(task_name: str) -> str:
    if task_name == "winogrande_v2:":
        return "winogrande"
    elif task_name == "hellaswag_v2:":
        return "hellaswag"
    elif "_MCQs:" in task_name:
        return "GaoKao"
    elif task_name.lower() in ["math", "psychology", "physics", "other", "economics"]:
        return "MMLU_Pro"
    else:
        return "BBH"
    

def get_results_by_subset(exp_result_path: str, eval_data_path: str, model_name_list: list, exp_num=8, force_read=True):
    all_summary_path = os.path.join(exp_result_path, "all_summary.json")
    if not force_read and os.path.exists(all_summary_path):
        with open(all_summary_path, "r", encoding='utf-8') as f:
            all_result_dict = json.load(f)
        return all_result_dict, None

    all_result_dict = defaultdict(lambda: defaultdict(list))
    exp_result_dict_list = []
    
    for exp_idx in tqdm(range(exp_num), ncols=100):
        cur_result_dict = defaultdict(lambda: defaultdict(list))
        cur_eval_data_path = os.path.join(eval_data_path, f"{exp_idx}.json")
        with open(cur_eval_data_path, "r", encoding='utf-8') as f:
           eval_data_list = json.load(f)

        for model_name in model_name_list:
            cur_exp_result_file = os.path.join(exp_result_path, model_name, str(exp_idx), "local_model_eval_result.json")
            with open(cur_exp_result_file, "r", encoding='utf-8') as f:
                cur_exp_result_list = json.load(f)
            assert len(cur_exp_result_list) == len(eval_data_list), f"Length not match for eval data and eval results!!! {len(eval_data_list)} v.s. {len(cur_exp_result_list)}"
            for idx in range(len(eval_data_list)):
                eval_data_record = eval_data_list[idx]
                cur_exp_result_record = cur_exp_result_list[idx]
                try:
                    is_correct = cur_exp_result_record["is_correct"]
                except KeyError:
                    is_correct = cur_exp_result_record["is_corrent"]
                subset_name = eval_data_record["full_name"]
                cur_result_dict[model_name][subset_name].append(is_correct)

        for model_name in cur_result_dict.keys():
            for subset_name in cur_result_dict[model_name].keys():
                subset_acc = np.mean(cur_result_dict[model_name][subset_name])
                all_result_dict[model_name][subset_name].append(subset_acc)
        
        exp_result_dict_list.append(dict(cur_result_dict))

    with open(all_summary_path, "w", encoding='utf-8') as f:
        json.dump(all_result_dict, f, ensure_ascii=False, indent=2)
        
    return dict(all_result_dict), exp_result_dict_list


def orp_calculation(model_A_results: list, model_B_results: list, sample_num=100, max_delta=1.0) -> float:
    value_list = []
    # 计算标准差
    std1 = np.std(model_A_results, ddof=1)
    std2 = np.std(model_B_results, ddof=1)
    # 计算皮尔逊相关系数
    corr, _ = pearsonr(model_A_results, model_B_results)
    # 计算delta的方差
    delta_variance = std1**2 + std2**2 - 2*corr*std1*std2
    # 初始化 ORP 值
    orp_value = 0.0
    # 积分步骤
    granularity = max_delta / sample_num
    for delta in np.arange(0, max_delta+granularity, granularity):
        cdf_value = norm.cdf(0, loc=delta, scale=np.sqrt(delta_variance))
        value_list.append(cdf_value)
        # orp_value += cdf_value * (max_delta / sample_num)
        orp_value += cdf_value * (1 / sample_num)
    return orp_value, value_list


def corr_analysis(all_result_dict: dict, fig_save_path_name: str, result_path: str,\
                   corr_type="pearson", draw_fig=args.draw_corr_fig, verbose=False):
    def _draw_corr_heatmap(corr_matrix, model_name_list, save_path, save_name):
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True,\
                     xticklabels=model_name_list, yticklabels=model_name_list, vmin=-1, vmax=1)
        plt.title(f'{corr_type} on {save_name}')
        os.makedirs(os.path.join(save_path, "jpg"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "pdf"), exist_ok=True)
        plt.savefig(os.path.join(save_path, "jpg", f"{save_name}.jpg"), dpi=100)
        plt.savefig(os.path.join(save_path, "pdf", f"{save_name}.pdf"))
        plt.close()

    assert corr_type in ["pearson", "spearman"], f"Correlation type should be in ['pearson', 'spearman'], but got {corr_type}"
    corr_func = pearsonr if corr_type == "pearson" else spearmanr
    fig_save_path = os.path.join(result_path, f"{fig_save_path_name}_{corr_type}")
    os.makedirs(fig_save_path, exist_ok=True)
    all_model_name = list(all_result_dict.keys())
    all_task_name = list(all_result_dict[all_model_name[0]].keys())
    all_corr_matrix = np.zeros((len(all_model_name), len(all_model_name)))
    valid_num = len(all_task_name)
    for task_name in tqdm(all_task_name, ncols=100):
        corr_matrix = np.zeros((len(all_model_name), len(all_model_name)))
        for i, model_name_A in enumerate(all_model_name):
            for j, model_name_B in enumerate(all_model_name):
                if i > j:
                    corr_matrix[i][j] = corr_matrix[j][i]
                    continue
                if i == j:
                    corr_matrix[i][j] = 1.0
                    continue
                corr_matrix[i][j] = corr_func(all_result_dict[model_name_A][task_name], all_result_dict[model_name_B][task_name])[0]
        if np.isnan(corr_matrix).any():
            if verbose:
                print(f"Got Nan value in task {task_name}, skip!!!")
            valid_num -= 1
            continue
        all_corr_matrix += corr_matrix
        if draw_fig:
            _draw_corr_heatmap(corr_matrix, all_model_name, save_path=fig_save_path, save_name=task_name)
    all_corr_matrix /= valid_num
    if draw_fig:
        print(f"Saving figs to {fig_save_path}")
        _draw_corr_heatmap(all_corr_matrix, all_model_name, save_path=fig_save_path, save_name="1_all_average")
    upper_tri_indices = np.triu_indices(all_corr_matrix.shape[0], k=1)
    orp_average_value = np.mean(all_corr_matrix[upper_tri_indices])
    return orp_average_value


def mean_std_analysis(all_result_dict, exp_result_dict_list, subset_size_dict, result_path, save_name):
    # 创建保存结果的文件夹
    save_dir = os.path.join(result_path, save_name)
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    models_data = {}  # 用于每个model_name单独保存结果
    
    for task_name in tqdm(all_result_dict[list(all_result_dict.keys())[0]].keys(), ncols=100):
        subset_size = subset_size_dict[task_name]
        for model_name in all_result_dict.keys():
            results = all_result_dict[model_name][task_name]
            mean = np.mean(results)
            std_dev = np.std(results)
            range_ = np.ptp(results)  # range is a keyword in Python
            
            # Identify experiments with max and min mean values
            max_mean_idx = np.argmax(results)
            min_mean_idx = np.argmin(results)
            
            # Extract instances result for McNemar's test
            max_experiment_results = exp_result_dict_list[max_mean_idx][model_name][task_name]
            min_experiment_results = exp_result_dict_list[min_mean_idx][model_name][task_name]
            
            # Create a contingency table for McNemar's test
            contingency_table = np.zeros((2, 2))
            for max_res, min_res in zip(max_experiment_results, min_experiment_results):
                contingency_table[int(max_res)][int(min_res)] += 1
            
            # Perform McNemar's test
            result = mcnemar(contingency_table)
            p_value = result.pvalue
            
            # Format the results to three decimal places
            formatted_results = ','.join(f'{r:.3f}' for r in results)
            
            row = {
                'task_name': task_name,
                'model_name': model_name,
                'results': formatted_results,
                'subset_size': subset_size,
                'mean': round(mean, 4),
                'std_dev': round(std_dev, 4),
                'range': round(range_, 4),
                'p_value': round(p_value, 4)
            }
            rows.append(row)
            if model_name not in models_data:
                models_data[model_name] = []
            models_data[model_name].append(row)
    
    # Create DataFrame for all models
    df = pd.DataFrame(rows)
    df = df[['task_name', 'model_name', 'results', 'subset_size', 'mean', 'std_dev', 'range', 'p_value']]
    
    # Save the combined DataFrame to an Excel file
    df.to_excel(os.path.join(save_dir, "model_task_detail.xlsx"), index=True, sheet_name='Sheet1')

    # Save each model's data to separate Excel files
    for model_name, model_rows in models_data.items():
        model_df = pd.DataFrame(model_rows)
        model_df = model_df[['task_name', 'results', 'subset_size', 'mean', 'std_dev', 'range', 'p_value']]  # 去掉'model_name'列
        model_df.to_excel(os.path.join(save_dir, f"{model_name}_task_detail.xlsx"), index=True, sheet_name='Sheet1')
    avg_mean, avg_std_dev, avg_range = df['mean'].mean(), df['std_dev'].mean(), df['range'].mean()
    return avg_mean, avg_std_dev, avg_range


def compare_benchmark_level_acc(all_result_dict_fixed, all_result_dict_random, save_root_path, save_name):
    # 创建保存结果的文件夹
    save_dir = os.path.join(save_root_path, save_name)
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    models_data = {}  # 用于每个model_name单独保存结果
    
    for task_name in tqdm(all_result_dict_fixed[list(all_result_dict_fixed.keys())[0]].keys(), ncols=100):
        for model_name in all_result_dict_fixed.keys():
            results_fixed = all_result_dict_fixed[model_name][task_name]
            mean_fixed = np.mean(results_fixed)
            std_dev_fixed = np.std(results_fixed)
            range_fixed = np.ptp(results_fixed)
            results_random = all_result_dict_random[model_name][task_name]
            mean_random = np.mean(results_random)
            std_dev_random = np.std(results_random)
            range_random = np.ptp(results_random)
            
            # Format the results to three decimal places
            formatted_results_fixed = ','.join(f'{r:.3f}' for r in results_fixed)
            formatted_results_random = ','.join(f'{r:.3f}' for r in results_random)
            
            row = {
                'task_name': task_name,
                'model_name': model_name,
                'results_fixed': formatted_results_fixed,
                'results_random': formatted_results_random,
                'mean_fixed': round(mean_fixed, 4),
                'mean_random': round(mean_random, 4),
                'std_dev_fixed': round(std_dev_fixed, 4),
                'std_dev_random': round(std_dev_random, 4),
                'range_fixed': round(range_fixed, 4),
                'range_random': round(range_random, 4),
            }
            rows.append(row)
            if model_name not in models_data:
                models_data[model_name] = []
            models_data[model_name].append(row)
    
    # Create DataFrame for all models
    df = pd.DataFrame(rows)
    df = df[['task_name', 'model_name', 'results_fixed', 'results_random', 'mean_fixed', 'mean_random', 'std_dev_fixed', 'std_dev_random', 'range_fixed', 'range_random']]
    
    # Save the combined DataFrame to an Excel file
    df.to_excel(os.path.join(save_dir, "model_task_detail_compare.xlsx"), index=True, sheet_name='Sheet1')

    # Save each model's data to separate Excel files
    for model_name, model_rows in models_data.items():
        model_df = pd.DataFrame(model_rows)
        model_df = model_df[['task_name', 'results_fixed', 'results_random', 'mean_fixed', 'mean_random', 'std_dev_fixed', 'std_dev_random', 'range_fixed', 'range_random']]  # 去掉'model_name'列
        model_df.to_excel(os.path.join(save_dir, f"{model_name}_task_detail_compare.xlsx"), index=True, sheet_name='Sheet1')


def orp_value_analysis(all_result_dict, subset_size_dict, result_path, save_name, sample_num=100, max_delta=1.0):
    # 创建文件夹
    save_dir = os.path.join(result_path, save_name)
    os.makedirs(save_dir, exist_ok=True)
    
    orp_detail = []
    orp_average = []
    orp_value_list = []
    
    # 遍历所有任务
    all_task_list = next(iter(all_result_dict.values())).keys()
    all_model_name_list = list(all_result_dict.keys())
    model_sbs_orp_matrix = np.zeros((len(all_model_name_list), len(all_model_name_list)))
    valid_num = len(all_task_list)

    for task_name in tqdm(all_task_list, ncols=100):  # 假定所有模型都有相同的task
        models = list(all_result_dict.keys())
        num_models = len(models)
        subset_size = subset_size_dict[task_name]
        orp_matrix = np.zeros((num_models, num_models))

        # 计算 ORP 矩阵
        for i in range(num_models):
            for j in range(i + 1, num_models):
                model_A_results = all_result_dict[models[i]][task_name]
                model_B_results = all_result_dict[models[j]][task_name]
                orp_value, _ = orp_calculation(model_A_results, model_B_results, sample_num=sample_num, max_delta=max_delta)
                orp_matrix[i, j] = orp_matrix[j, i] = orp_value
        if np.isnan(orp_matrix).any():
            valid_num -= 1
        else:
            model_sbs_orp_matrix += orp_matrix
        
        # 保存 ORP 详细信息
        for i in range(num_models):
            orp_detail_entry = [task_name, models[i]]
            orp_detail_entry += [round(orp_matrix[i, j], 5) for j in range(num_models)]
            orp_detail.append(orp_detail_entry)
        
        # 计算并保存 ORP 平均值
        upper_tri_indices = np.triu_indices(num_models, k=1)
        orp_average_value = np.mean(orp_matrix[upper_tri_indices])
        orp_average.append([task_name, round(orp_average_value, 5), subset_size])
        if not np.isnan(orp_average_value):
            orp_value_list.append(orp_average_value)

    # 创建 ORP 详细信息的 DataFrame
    column_names = ["task_name", "model_name"] + models
    orp_detail_df = pd.DataFrame(orp_detail, columns=column_names)
    orp_detail_filepath = os.path.join(save_dir, "orp_detail.xlsx")
    orp_detail_df.to_excel(orp_detail_filepath, index=False)
    
    # 创建 ORP 平均值的 DataFrame
    orp_average_df = pd.DataFrame(orp_average, columns=["task_name", "average_orp", "data_size"])
    orp_average_filepath = os.path.join(save_dir, "orp_average.xlsx")
    orp_average_df.to_excel(orp_average_filepath, index=False)

    # 创建 ORP 模型两两间在数据子集层面平均的 DataFrame
    model_sbs_orp_matrix /= valid_num
    model_sbs_orp_df = pd.DataFrame(model_sbs_orp_matrix, columns=all_model_name_list, index=all_model_name_list)
    model_sbs_orp_filepath = os.path.join(save_dir, "model_sbs_orp.xlsx")
    model_sbs_orp_df.to_excel(model_sbs_orp_filepath, index=False)

    return np.mean(orp_value_list)


def draw_orp_curves(exp_result_dict_list, exp_name_list, save_root_path, save_name, draw_name,\
                     sample_num=50, max_delta=1.0, confidence_val_list=[0.9, 0.95, 0.99], use_marker=True):
    def _normalize_exp_name(exp_name: str) -> str:
        return exp_name.replace("fixed_", "fewshot_").replace("fix_", "").replace("random_", "fewshot_").replace("rand_", "")

    fig_save_path = os.path.join(save_root_path, save_name)
    os.makedirs(fig_save_path, exist_ok=True)
    model_name_list = list(exp_result_dict_list[0].keys())
    task_name_list = list(next(iter(exp_result_dict_list[0].values())).keys())
    all_markers = ['o', '^', 's', '*', 'D', 'X', 'P']
    x = np.arange(0, max_delta+max_delta/sample_num, max_delta/sample_num)
    true_name_list = list(set([_normalize_exp_name(x) for x in exp_name_list]))
    cmap = plt.get_cmap("viridis")
    confidence_interval_record = {}
    for idx, (exp_name, all_result_dict) in enumerate(zip(exp_name_list, exp_result_dict_list)):
        all_orp_values = np.zeros(sample_num + 1)
        all_valid_num = 0
        for task_name in tqdm(task_name_list, ncols=100):
            task_orp_values = np.zeros(sample_num + 1)
            cur_valid_num = 0
            for model_name_A in model_name_list:
                for model_name_B in model_name_list:
                    model_A_results = all_result_dict[model_name_A][task_name]
                    model_B_results = all_result_dict[model_name_B][task_name]
                    _, orp_value_list = orp_calculation(model_A_results, model_B_results, sample_num=sample_num, max_delta=max_delta)
                    if not np.isnan(orp_value_list).any():
                        task_orp_values += np.array(orp_value_list)
                        cur_valid_num += 1
            task_orp_values /= cur_valid_num
            if cur_valid_num > 0:
                all_orp_values += task_orp_values
                all_valid_num += 1
        all_orp_values /= all_valid_num
        color_value = min(1.0, idx/len(exp_name_list)+0.1) if len(exp_name_list) <= 2 else\
              min(1.0, true_name_list.index(_normalize_exp_name(exp_name))/len(true_name_list)+0.1)
        marker_value = all_markers[0 if "rand" in exp_name else 1]
        if use_marker:
            plt.plot(x, all_orp_values, label=f"{exp_name}:{np.mean(all_orp_values):.4f}", color=cmap(color_value),\
                  marker=marker_value, markersize=3, markevery=[0, len(all_orp_values)//2, len(all_orp_values)-1])
        else:
            plt.plot(x, all_orp_values, label=f"{exp_name}:{np.mean(all_orp_values):.4f}", color=cmap(color_value))
        granularity = max_delta / sample_num
        con_idx, confidence_delta_record = 0, {}
        for sample_idx, orp_val in enumerate(all_orp_values):
            con_val = confidence_val_list[con_idx]
            if orp_val <= 1 - con_val:
                confidence_delta_record[con_val] = sample_idx * granularity
                con_idx += 1
            if con_idx >= len(confidence_val_list):
                break
        confidence_interval_record[exp_name] = confidence_delta_record
    
    for con_idx, con_val in enumerate(confidence_val_list):
        con_color = min(1.0, con_idx/len(confidence_val_list)+0.1)
        plt.axhline(y=1-con_val, color=cmap(con_color), alpha=0.5)
    print(json.dumps(confidence_interval_record, indent=2, ensure_ascii=False))
    plt.legend(fontsize=5)
    plt.grid(True, linestyle='--')
    plt.savefig(os.path.join(fig_save_path, f"{draw_name}.jpg"), dpi=150)
    plt.close()


def draw_orp_curves_split(exp_result_dict_list, exp_name_list, save_root_path, save_name, draw_name, task_name_filter,\
                     sample_num=50, max_delta=1.0, confidence_val_list=[0.9, 0.95, 0.99]):
    def _normalize_exp_name(exp_name: str) -> str:
        return exp_name
        # return exp_name.replace("fixed_", "fewshot_").replace("fix_", "").replace("random_", "fewshot_").replace("rand_", "")

    fig_save_path = os.path.join(save_root_path, save_name)
    os.makedirs(fig_save_path, exist_ok=True)
    model_name_list = list(exp_result_dict_list[0].keys())
    task_name_list = list(next(iter(exp_result_dict_list[0].values())).keys())
    x = np.arange(0, max_delta+max_delta/sample_num, max_delta/sample_num)
    true_name_list = list(set([_normalize_exp_name(x) for x in exp_name_list]))
    cmap = plt.get_cmap("viridis")
    confidence_interval_record = {}
    
    for dataset_name in ["MMLU_Pro"] if args.data_version == "mmlu_pro" else ["winogrande", "hellaswag", "GaoKao", "BBH"]:
        plt.figure(figsize=(6, 4))
        for idx, (exp_name, all_result_dict) in enumerate(zip(exp_name_list, exp_result_dict_list)):
            all_orp_values = np.zeros(sample_num + 1)
            all_valid_num = 0
            for task_name in tqdm(task_name_list, ncols=100):
                if task_name_filter(task_name) != dataset_name:
                    continue
                task_orp_values = np.zeros(sample_num + 1)
                cur_valid_num = 0
                for model_name_A in model_name_list:
                    for model_name_B in model_name_list:
                        model_A_results = all_result_dict[model_name_A][task_name]
                        model_B_results = all_result_dict[model_name_B][task_name]
                        _, orp_value_list = orp_calculation(model_A_results, model_B_results, sample_num=sample_num, max_delta=max_delta)
                        if not np.isnan(orp_value_list).any():
                            task_orp_values += np.array(orp_value_list)
                            cur_valid_num += 1
                task_orp_values /= cur_valid_num
                if cur_valid_num > 0:
                    all_orp_values += task_orp_values
                    all_valid_num += 1

            all_orp_values /= all_valid_num
            color_value = min(1.0, idx/len(exp_name_list)+0.1) if len(exp_name_list) <= 2 else\
                min(1.0, true_name_list.index(_normalize_exp_name(exp_name))/len(true_name_list)+0.1)
            plt.plot(x, all_orp_values, lw=3, label=f"{exp_name}:{np.mean(all_orp_values):.4f}", color=cmap(color_value))
            granularity = max_delta / sample_num
            con_idx, confidence_delta_record = 0, {}
            for sample_idx, orp_val in enumerate(all_orp_values):
                con_val = confidence_val_list[con_idx]
                if orp_val <= 1 - con_val:
                    confidence_delta_record[con_val] = sample_idx * granularity
                    con_idx += 1
                if con_idx >= len(confidence_val_list):
                    break
            confidence_interval_record[exp_name] = confidence_delta_record
        
        for con_idx, con_val in enumerate(confidence_val_list):
            con_color = min(1.0, con_idx/len(confidence_val_list)+0.1)
            plt.axhline(y=1-con_val, color=cmap(con_color), alpha=0.5, lw=2.5)
        print(json.dumps(confidence_interval_record, indent=2, ensure_ascii=False))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('True Difference of Model Performance', fontsize=16)
        plt.ylabel('ORP Value', fontsize=16)
        plt.title(dataset_name.title() if dataset_name not in ["GaoKao", "BBH", "MMLU_Pro"] else dataset_name, fontsize=17)
        plt.legend(fontsize=15)
        plt.grid(True, linestyle='--')
        # plt.subplots_adjust(left=0.05, bottom=0.05, right=0.15, top=0.15)
        plt.tight_layout()
        real_save_path = os.path.join(fig_save_path, dataset_name)
        os.makedirs(real_save_path, exist_ok=True)
        plt.savefig(os.path.join(real_save_path, f"{draw_name}.pdf"))
        plt.close()


def compare_with_biased(biased_orp_path, exp_orp_path, result_save_path, save_name):
    save_path = os.path.join(result_save_path, "orp_comparison")
    os.makedirs(save_path, exist_ok=True)
    biased_orp_df = pd.read_excel(biased_orp_path, engine='openpyxl')
    exp_orp_df = pd.read_excel(exp_orp_path, engine='openpyxl')
    biased_orp_df.rename(columns={'average_orp': 'biased_orp'}, inplace=True)
    exp_orp_df.rename(columns={'average_orp': 'unbiased_orp'}, inplace=True)
    merged_df = pd.merge(biased_orp_df, exp_orp_df, on='task_name')
    merged_df['diff'] = merged_df['biased_orp'] - merged_df['unbiased_orp']
    merged_df['diff'].fillna(0, inplace=True)
    merged_df.to_excel(os.path.join(save_path, f"{save_name}.xlsx"), index=False)
    # 绘制diff的变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(merged_df) + 1), sorted(merged_df["diff"]), color='#1f77b4', label='Diff')  # 柔和的蓝色
    plt.axhline(y=0, color='#ff7f0e', linestyle='--', label='y=0')  # 柔和的红色
    plt.xlabel('Task Index')
    plt.ylabel('Diff')
    plt.title(f'{save_name} ORP Diff (larger better)')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig(os.path.join(save_path, f"{save_name}.jpg"), dpi=150)
    plt.close()
