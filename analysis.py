from utils import *


MODEL_NAME_LIST = ["glm4-9B", "llama3-8B", "gemma2-9B", "qwen2.5-7B"] if args.data_version == "mmlu_pro" else ["glm4-9B", "llama3-8B", "llama3-70B", "qwen2-72B"]


def single_setting_analysis_pipeline(exp_result_path, eval_data_path, model_name_list, analysis_name="analysis", exp_num=8,\
                                     fig_save_path_name="corr_figs", statistic_save_name="statistics_analysis",\
                                     orp_save_name="orp_value_analysis", max_delta=1.0, draw_fig=args.draw_corr_fig, verbose=False, force_read=False):
    # 读取结果
    print("Reading results...")
    all_result_dict, exp_result_dict_list = get_results_by_subset(exp_result_path, eval_data_path, model_name_list, exp_num=exp_num, force_read=force_read)
    subset_size_dict = {task_name: len(task_result_list) for task_name, task_result_list in exp_result_dict_list[0][model_name_list[0]].items()}
    exp_result_path = os.path.join(exp_result_path, analysis_name)
    # 统计指标分析
    print("Statistic analysis...")
    avg_mean, avg_std_dev, avg_range = mean_std_analysis(all_result_dict, exp_result_dict_list, subset_size_dict, result_path=exp_result_path, save_name=statistic_save_name)
    print(f"Average  mean: {avg_mean:.5f}, std_dev: {avg_std_dev:.5f}, range: {avg_range:.5f}")
    # ORP值分析
    print("ORP value analysis...")
    average_orp_value = orp_value_analysis(all_result_dict, subset_size_dict, result_path=exp_result_path, save_name=orp_save_name, max_delta=max_delta)
    print(f"Average ORP value: {average_orp_value:.5f}")
    # 相关性分析
    print("Correlation analysis...")
    for corr_type in ["pearson", "spearman"]:
        print(f"Type: {corr_type}")
        orp_average_value = corr_analysis(all_result_dict, fig_save_path_name, result_path=exp_result_path, corr_type=corr_type, draw_fig=draw_fig, verbose=verbose)
        print(f"Average {corr_type} value: {orp_average_value:.5f}")


def compare_analysis_pipeline(fixed_result_path, unbiased_result_path_list, eval_data_path, model_name_list,\
                               exp_num=8, max_delta=1.0, force_read=False, analysis_name="analysis",\
                               analysis_root="comparison_analysis", draw_name=None, draw_orp_dataset_num=False):
    all_result_dict_list = []
    exp_result_path_list = [fixed_result_path] + unbiased_result_path_list
    if draw_name is None:
        exp_name_list = [os.path.basename(exp_result_path) for exp_result_path in exp_result_path_list]
    else:
        exp_name_list = [x.strip() for x in draw_name.split("v.s.")]
    result_root_path = os.path.join(os.path.dirname(exp_result_path_list[0]), analysis_root)
    os.makedirs(result_root_path, exist_ok=True)

    print("Compare statictic results of biased and unbiased results...")
    all_result_dict_fixed, _ = get_results_by_subset(fixed_result_path, eval_data_path, model_name_list, exp_num=exp_num, force_read=force_read)
    biased_name = os.path.basename(fixed_result_path)
    for unbiased_result_path in unbiased_result_path_list:
        all_result_dict_random, _ = get_results_by_subset(unbiased_result_path, eval_data_path, model_name_list, exp_num=exp_num, force_read=force_read)
        unbiased_name = os.path.basename(unbiased_result_path)
        compare_benchmark_level_acc(all_result_dict_fixed, all_result_dict_random, save_root_path=result_root_path, save_name=f"{biased_name}###{unbiased_name}")

    print("Draw ORP curves...")
    for exp_result_path in exp_result_path_list:
        # 读取结果
        print(f"Reading results of {exp_result_path}...")
        cur_all_result_dict, _ = get_results_by_subset(exp_result_path, eval_data_path, model_name_list, exp_num=exp_num, force_read=force_read)
        all_result_dict_list.append(cur_all_result_dict)
    if draw_name is None:
        draw_name = "###".join(exp_name_list) if len(unbiased_result_path_list) <= 2 else "###".join([x.replace("rand_", "").replace("random", "") for x in exp_name_list if "rand" in x])

    draw_orp_curves_split(all_result_dict_list, exp_name_list, save_root_path=result_root_path,\
                            task_name_filter=task_name_func, save_name="orp_curves", draw_name=draw_name, max_delta=max_delta)

    if draw_orp_dataset_num:
        print("Comparing fixed & unbiased orp results...")
        biased_orp_path = os.path.join(fixed_result_path, analysis_name, "orp_value_analysis", "orp_average.xlsx")
        biased_name = os.path.basename(fixed_result_path)
        assert os.path.exists(biased_orp_path), f"Single exp result not run! Therefore can not find path {biased_orp_path}. Should run first!!!"
        for unbiased_result_path in unbiased_result_path_list:
            unbiased_name = os.path.basename(unbiased_result_path)
            unbiased_analysis_name = os.path.basename(unbiased_result_path)
            if unbiased_analysis_name == "rand_fewshots":
                unbiased_analysis_name = "rand_fewshots"
            unbiased_orp_path = os.path.join(unbiased_result_path, unbiased_analysis_name, "orp_value_analysis", "orp_average.xlsx")
            assert os.path.exists(unbiased_orp_path),  f"Single exp result not run! Therefore can not find path {unbiased_orp_path}. Should run first!!!"
            compare_with_biased(biased_orp_path, unbiased_orp_path, result_save_path=result_root_path, save_name=f"{biased_name}@{unbiased_name}")
    

if __name__ == '__main__':
    # %% [markdown]
    # # Fewshot设定：

    # %% [markdown]
    # ## 所有随机因素

    # %% [markdown]
    # ### 固定结果分析：

    # %%
    exp_result_path = f"./results/{args.data_version}/fix_fewshots@prompt_format@task_description@question_prefix@option_label"
    eval_data_path = f"./data/{args.data_version}/fix_fewshots@prompt_format@task_description@question_prefix@option_label"
    fixed_analysis_name = "fix_fewshots@prompt_format@task_description@question_prefix@option_label"
    max_delta = 0.3
    single_setting_analysis_pipeline(exp_result_path, eval_data_path, MODEL_NAME_LIST, analysis_name=fixed_analysis_name, draw_fig=args.draw_corr_fig, max_delta=max_delta, force_read=args.force_read)

    # %% [markdown]
    # ### 随机化结果分析：

    # %%
    exp_result_path = f"./results/{args.data_version}/rand_fewshots@prompt_format@task_description@question_prefix@option_label"
    eval_data_path = f"./data/{args.data_version}/rand_fewshots@prompt_format@task_description@question_prefix@option_label"
    analysis_name = "rand_fewshots@prompt_format@task_description@question_prefix@option_label"
    max_delta = 0.3
    single_setting_analysis_pipeline(exp_result_path, eval_data_path, MODEL_NAME_LIST, analysis_name=analysis_name, draw_fig=args.draw_corr_fig, max_delta=max_delta, force_read=args.force_read)

    # %% [markdown]
    # ### 消偏方法对比分析：

    # %%
    fixed_result_path = f"./results/{args.data_version}/fix_fewshots@prompt_format@task_description@question_prefix@option_label"
    unbiased_result_path_list = [
        f"./results/{args.data_version}/rand_fewshots@prompt_format@task_description@question_prefix@option_label",
    ]
    eval_data_path = f"./data/{args.data_version}/fix_fewshots@prompt_format@task_description@question_prefix@option_label"
    draw_name = "fixed_all v.s. random_all"
    compare_analysis_pipeline(fixed_result_path, unbiased_result_path_list, eval_data_path, MODEL_NAME_LIST, draw_name=draw_name, analysis_name=fixed_analysis_name, max_delta=0.4, force_read=args.force_read)

    # %% [markdown]
    # ## Task Description

    # %% [markdown]
    # ### 固定setting：

    # %%
    exp_result_path = f"./results/{args.data_version}/fix_task_description@question_prefix"
    eval_data_path = f"./data/{args.data_version}/fix_task_description@question_prefix"
    fixed_analysis_name = "fixed_task_description@question_prefix"
    max_delta = 0.3
    single_setting_analysis_pipeline(exp_result_path, eval_data_path, MODEL_NAME_LIST, analysis_name=fixed_analysis_name, draw_fig=args.draw_corr_fig, max_delta=max_delta, force_read=args.force_read)

    # %% [markdown]
    # ### 随机化setting：

    # %%
    exp_result_path = f"./results/{args.data_version}/rand_task_description@question_prefix"
    eval_data_path = f"./data/{args.data_version}/rand_task_description@question_prefix"
    analysis_name = "random_task_description@question_prefix"
    max_delta = 0.3
    single_setting_analysis_pipeline(exp_result_path, eval_data_path, MODEL_NAME_LIST, analysis_name=analysis_name, draw_fig=args.draw_corr_fig, max_delta=max_delta, force_read=args.force_read)

    # %% [markdown]
    # ### 消偏方法对比分析：

    # %%
    fixed_result_path = f"./results/{args.data_version}/fix_task_description@question_prefix"
    unbiased_result_path_list = [
        f"./results/{args.data_version}/rand_task_description@question_prefix",
    ]
    eval_data_path = f"./data/{args.data_version}/fix_task_description@question_prefix"
    draw_name = "fixed_task_description v.s. random_task_description"
    compare_analysis_pipeline(fixed_result_path, unbiased_result_path_list, eval_data_path, MODEL_NAME_LIST, draw_name=draw_name, analysis_name=fixed_analysis_name, max_delta=0.4, force_read=args.force_read)

    # %% [markdown]
    # ## Option Label

    # %% [markdown]
    # ### 固定setting：

    # %%
    exp_result_path = f"./results/{args.data_version}/fix_option_label"
    eval_data_path = f"./data/{args.data_version}/fix_option_label"
    fixed_analysis_name = "fix_option_label"
    max_delta = 0.3
    single_setting_analysis_pipeline(exp_result_path, eval_data_path, MODEL_NAME_LIST, analysis_name=fixed_analysis_name, draw_fig=args.draw_corr_fig, max_delta=max_delta, force_read=args.force_read)

    # %% [markdown]
    # ### 随机化setting：

    # %%
    exp_result_path = f"./results/{args.data_version}/rand_option_label"
    eval_data_path = f"./data/{args.data_version}/rand_option_label"
    analysis_name = "rand_option_label"
    max_delta = 0.3
    single_setting_analysis_pipeline(exp_result_path, eval_data_path, MODEL_NAME_LIST, analysis_name=analysis_name, draw_fig=args.draw_corr_fig, max_delta=max_delta, force_read=args.force_read)

    # %% [markdown]
    # ### 消偏方法对比分析：

    # %%
    fixed_result_path = f"./results/{args.data_version}/fix_option_label"
    unbiased_result_path_list = [
        f"./results/{args.data_version}/rand_option_label",
    ]
    eval_data_path = f"./data/{args.data_version}/fix_option_label"
    draw_name = "fixed_option_label v.s. random_option_label"
    compare_analysis_pipeline(fixed_result_path, unbiased_result_path_list, eval_data_path, MODEL_NAME_LIST, draw_name=draw_name, analysis_name=fixed_analysis_name, max_delta=0.4, force_read=args.force_read)

    # %% [markdown]
    # ## fewshot examples

    # %% [markdown]
    # ### 固定setting：

    # %%
    exp_result_path = f"./results/{args.data_version}/fix_fewshots"
    eval_data_path = f"./data/{args.data_version}/fix_fewshots"
    fixed_analysis_name = "fix_fewshots"
    max_delta = 0.3
    single_setting_analysis_pipeline(exp_result_path, eval_data_path, MODEL_NAME_LIST, analysis_name=fixed_analysis_name, draw_fig=args.draw_corr_fig, max_delta=max_delta, force_read=args.force_read)

    # %% [markdown]
    # ### 随机化setting：

    # %%
    exp_result_path = f"./results/{args.data_version}/rand_fewshots"
    eval_data_path = f"./data/{args.data_version}/rand_fewshots"
    analysis_name = "rand_fewshots"
    max_delta = 0.3
    single_setting_analysis_pipeline(exp_result_path, eval_data_path, MODEL_NAME_LIST, analysis_name=analysis_name, draw_fig=args.draw_corr_fig, max_delta=max_delta, force_read=args.force_read)

    # %% [markdown]
    # ### 消偏方法对比分析：

    # %%
    fixed_result_path = f"./results/{args.data_version}/fix_fewshots"
    unbiased_result_path_list = [
        f"./results/{args.data_version}/rand_fewshots",
    ]
    eval_data_path = f"./data/{args.data_version}/fix_fewshots"
    draw_name = "fix_fewshots v.s. rand_fewshots"
    compare_analysis_pipeline(fixed_result_path, unbiased_result_path_list, eval_data_path, MODEL_NAME_LIST, draw_name=draw_name, analysis_name=fixed_analysis_name, max_delta=0.4, force_read=args.force_read)

    # %% [markdown]
    # ## Prompt Format:

    # %% [markdown]
    # ### 固定结果分析：

    # %%
    if args.data_version == "mmlu_pro":
        exp_result_path = f"./results/{args.data_version}/fix_prompt_format"
        eval_data_path = f"./data/{args.data_version}/fix_prompt_format"
        fixed_analysis_name = "fixed_prompt_format"
        max_delta = 0.3
        single_setting_analysis_pipeline(exp_result_path, eval_data_path, MODEL_NAME_LIST, analysis_name=fixed_analysis_name, draw_fig=args.draw_corr_fig, max_delta=max_delta, force_read=args.force_read)

    # %% [markdown]
    # ### 随机结果分析：

    # %%
    exp_result_path = f"./results/{args.data_version}/rand_prompt_format"
    eval_data_path = f"./data/{args.data_version}/rand_prompt_format"
    analysis_name = "random_prompt_format"
    max_delta = 0.3
    single_setting_analysis_pipeline(exp_result_path, eval_data_path, MODEL_NAME_LIST, analysis_name=analysis_name, draw_fig=args.draw_corr_fig, max_delta=max_delta, force_read=args.force_read)

    # %% [markdown]
    # ### 消偏方法对比分析：

    # %%
    fixed_result_path = f"./results/{args.data_version}/fix_prompt_format"
    unbiased_result_path_list = [
        f"./results/{args.data_version}/rand_prompt_format",
    ]
    eval_data_path = f"./data/{args.data_version}/fix_prompt_format"
    draw_name = "fixed_prompt_format v.s. random_prompt_format"
    compare_analysis_pipeline(fixed_result_path, unbiased_result_path_list, eval_data_path, MODEL_NAME_LIST, draw_name=draw_name, analysis_name=fixed_analysis_name, max_delta=0.4, force_read=args.force_read)


