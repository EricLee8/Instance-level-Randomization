# Instance-level Randomization

This repository contains the code and data for our paper at Findings of EMNLP 2025: Instance-level Randomization: Toward More Stable LLM Evaluations.

## Introduction
![A brief illustration of our ILR method. Different Roman numerals in circles represent different evaluation settings applied to each instance of a benchmark. The plus and minus signs mean that a certain evaluation setting has positive or negative effect on the evaluation result of an instance for a specific model. Different colors of the signs represent different models, who have specific preferences over certain evaluation settings.](./figures/main_fig-1.png)


**Problem:** Evaluating Large Language Models (LLMs) is unstable. Minor changes in random factors like few-shot examples, prompt formats, or task descriptions can drastically change evaluation scores and even reorder model rankings. Different models also have different preferences for these factors, making evaluations with a single fixed setting potentially unfair.

**Method:** To fix this, the paper proposes **Instance-Level Randomization (ILR)**. Instead of using one fixed setting (e.g., the same set of few-shot examples) for the entire benchmark in a single test run, ILR applies a randomly chosen setting to every single instance within the benchmark. The final score is the average from multiple such runs.

**Key Results:**
* **Faster & Cheaper Stability:** ILR makes the evaluation process more robust with significantly less computational cost. It can achieve the same level of stability (i.e., low variance) as previous methods using less than half the number of experimental runs.
* **Reduced Variance and Correlation:** Theoretical analysis and experiments show that ILR works by reducing the correlations between different instances within a benchmark and between different experimental runs.
* **Fairer and More Stable Rankings:** The paper introduces a new metric called **Observed Reversal Probability (ORP)** to measure how likely it is for model rankings to flip due to random factors. Experiments demonstrate that ILR significantly reduces the ORP, leading to more reliable and fair comparisons between models.

In short, by randomizing settings for each question rather than for the whole test, ILR provides a much more stable, fair, and efficient way to evaluate LLMs.


## Environment Setup
1. Create a conda environment:
    
    ```bash
    conda create -n ilr python=3.12
    conda activate ilr
    ```
1. Install the requirements:
    ```bash
    pip3 install -r requirements.txt
    ```

## Usage
1. unzip the `open_source.zip` to `./data/mmlu_pro` and `./data/public_CoT`.
1. unzip the `mmlu_pro_results.zip` to `./results/mmlu_pro`, where we provide the model outputs on MMLU-Pro as examples.
1. Use your evaluation framework to evaluate the data and output the model outputs as exampled in `./results`.
1. Run the analysis script:
    
    ```bash
    python3 analysis.py --data_version mmlu_pro --draw_corr_fig 0
    ```
    After running the script, analysis tables and figures will be generated in the same folder of the model outputs.
