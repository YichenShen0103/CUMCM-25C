import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sksurv.tree import SurvivalTree
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from graphviz import Digraph
import scipy.stats as stats
from scipy.stats import norm
from sksurv.metrics import concordance_index_censored
from lifelines.statistics import multivariate_logrank_test
from sksurv.util import Surv
warnings.filterwarnings("ignore")
data = pd.read_excel("data/data.xlsx", sheet_name=0)
features = ["年龄", "体重", "IVF妊娠", "孕妇BMI", "怀孕次数", "生产次数"]
feature_names_english = [
    "Age",
    "Weight",
    "IVF_Pregnancy",
    "BMI",
    "Pregnancy_Count",
    "Delivery_Count",
]
target = "Y染色体浓度"
data = data[features + [target]]
data.dropna(inplace=True)
X = data[features]
y = data[target]
X["IVF妊娠"] = X["IVF妊娠"].map(
    {"自然受孕": 0, "IVF（试管婴儿）": 1, "IUI（人工授精）": 1}
)
X["怀孕次数"] = X["怀孕次数"].replace("≥3", 3).astype(int)
X = X.fillna(X.mean(numeric_only=True))
valid_idx = y.notna()
X = X[valid_idx]
y = y[valid_idx]
print(f"Data shape after cleaning: {X.shape}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names_english)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
feature_importance = pd.DataFrame(
    {"features": feature_names_english, "importance": rf_model.feature_importances_}
)
feature_importance = feature_importance.sort_values(by="importance", ascending=False)
print("\nTop 5 key factors affecting Y chromosome concentration:")
print(feature_importance.head(5))
plt.figure(figsize=(10, 6))
plt.bar(
    feature_importance["features"], feature_importance["importance"], color="skyblue"
)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance for Y Chromosome Concentration")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.close()
corr_matrix = X_scaled.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    xticklabels=feature_names_english,
    yticklabels=feature_names_english,
)
plt.title("Correlation Matrix of Features")
plt.tight_layout()
plt.show()
plt.close()
vif_data = pd.DataFrame()
vif_data["Feature"] = feature_names_english
vif_data["VIF"] = [
    variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])
]
print("\nVariance Inflation Factor (VIF):")
print(vif_data.sort_values(by="VIF", ascending=False))
def week_to_numeric(week_str):
    try:
        week_str = str(week_str).replace(" ", "")
        week, days = (
            week_str.split("w+")
            if "w+" in week_str
            else (week_str.replace("w", ""), "0")
        )
        return float(week) + float(days) / 7
    except:
        return np.nan
try:
    data = pd.read_excel("data/data.xlsx", sheet_name=0)
except FileNotFoundError:
    print("Error: 'data/data.xlsx' file not found.")
    exit()
data["Gestational_Week_Numeric"] = data["检测孕周"].apply(week_to_numeric)
def get_target_time(group):
    target = group[group["Y染色体浓度"] >= 0.04]["Gestational_Week_Numeric"]
    if len(target) > 0:
        return pd.Series(
            {
                "Time": target.iloc[0],
                "Event": 1,
                "Retry_Count": len(group) - 1,
                "BMI": group["孕妇BMI"].iloc[0],
                "Weight": group["体重"].iloc[0],
                "Age": group["年龄"].iloc[0],
                "Pregnancy_Count": (
                    "3"
                    if group["怀孕次数"].iloc[0] == "≥3"
                    else group["怀孕次数"].iloc[0]
                ),
            }
        )
    else:
        return pd.Series(
            {
                "Time": group["Gestational_Week_Numeric"].max(),
                "Event": 0,
                "Retry_Count": len(group),
                "BMI": group["孕妇BMI"].iloc[0],
                "Weight": group["体重"].iloc[0],
                "Age": group["年龄"].iloc[0],
                "Pregnancy_Count": (
                    "3"
                    if group["怀孕次数"].iloc[0] == "≥3"
                    else group["怀孕次数"].iloc[0]
                ),
            }
        )
survival_data = (
    data.groupby("孕妇代码").apply(get_target_time, include_groups=False).reset_index()
)
survival_data[["BMI", "Weight", "Age", "Pregnancy_Count"]] = survival_data[
    ["BMI", "Weight", "Age", "Pregnancy_Count"]
].fillna(
    survival_data[["BMI", "Weight", "Age", "Pregnancy_Count"]].mean(numeric_only=True)
)
survival_data["Event"] = survival_data["Event"].astype(bool)
y = Surv.from_dataframe("Event", "Time", survival_data)
X = survival_data[["BMI", "Weight", "Age"]]
X["BMI"] = X["BMI"].astype(np.float32)
X["Weight"] = X["Weight"].astype(np.float32)
X["Age"] = X["Age"].astype(np.float32)
X = X.to_numpy()
tree = SurvivalTree(max_leaf_nodes=4, min_samples_leaf=25)  # 你可以调节 max_depth
tree.fit(X, y)
def print_survival_tree(tree, feature_names):
    """
    打印 SurvivalTree 分裂规则（修正版，支持叶子节点）
    """
    tree_ = tree.tree_
    left = tree_.children_left
    right = tree_.children_right
    threshold = tree_.threshold
    features = tree_.feature
    n_node_samples = tree_.n_node_samples
    def recurse(node, depth=0):
        indent = "  " * depth
        if left[node] == -1 and right[node] == -1:
            print(f"{indent}Leaf node: n_samples={n_node_samples[node]}")
        else:
            feature_index = features[node]
            print(
                f"{indent}{feature_names[feature_index]} <= {threshold[node]:.4f} (n_samples={n_node_samples[node]})"
            )
            recurse(left[node], depth + 1)
            recurse(right[node], depth + 1)
    recurse(0)
feature_names = ["BMI", "Weight", "Age"]
print_survival_tree(tree, feature_names)
leaf_id = tree.apply(X)
survival_data["leaf"] = leaf_id
plt.figure(figsize=(8, 6))
for leaf in np.unique(leaf_id):
    mask = leaf_id == leaf
    times, survival_prob = kaplan_meier_estimator(
        survival_data.loc[mask, "Event"], survival_data.loc[mask, "Time"]
    )
    plt.step(times, survival_prob, where="post", label=f"Leaf {leaf} (n={mask.sum()})")
plt.xlabel("Time")
plt.ylabel("Survival probability")
plt.legend()
plt.show()
event = survival_data["Event"].values
time = survival_data["Time"].values
risk_score = tree.predict(X)  # survival tree 输出的风险预测
c_index = concordance_index_censored(event, time, risk_score)
print("C-index:", c_index[0])
event = survival_data["Event"].values
time = survival_data["Time"].values
risk_score = tree.predict(X)  # survival tree 输出的风险预测
dot = Digraph()
dot.node("0", "Weight <= 95.75\n(n=267)")
dot.node("1", "Age <= 27.5\n(n=242)")
dot.node("2", "BMI <= 31.9651\n(n=84)")
dot.node("3", "Leaf\n(n=54)")
dot.node("4", "Leaf\n(n=30)")
dot.node("5", "Leaf\n(n=158)")
dot.node("6", "Leaf\n(n=25)")
dot.edge("0", "1", "Yes")
dot.edge("0", "6", "No")
dot.edge("1", "2", "Yes")
dot.edge("1", "5", "No")
dot.edge("2", "3", "Yes")
dot.edge("2", "4", "No")
def risk_penalty(t):
    """风险惩罚函数：较晚检测的风险增加"""
    return (t**2 - 1.0) / (624.0)  # 归一化到 [0, 1]
def risk_aware_optimal_time(group_data, alpha=1.0, beta=1.0, gamma=0.02):
    """
    风险感知的最优时间计算
    参数:
    - group_data: 组内数据
    - alpha: 失败率权重
    - beta: 风险惩罚权重
    - gamma: 时间偏好权重（鼓励更早检测）
    """
    from lifelines import KaplanMeierFitter
    if len(group_data) < 5:
        return {
            "optimal_time": np.nan,
            "survival_prob": np.nan,
            "coverage": np.nan,
            "objective": np.inf,
            "note": "Insufficient sample size",
        }
    kmf = KaplanMeierFitter()
    kmf.fit(group_data["Time"], event_observed=group_data["Event"])
    t_min = max(8.0, group_data["Time"].min())
    t_max = min(30.0, group_data["Time"].max())
    grid = np.linspace(t_min, t_max, 100)
    best_result = None
    best_objective = np.inf
    for t in grid:
        S_t = kmf.predict(t)
        coverage = 1.0 - S_t  # 检测成功率
        penalty = risk_penalty(t)  # 风险惩罚
        objective = alpha * (1 - coverage) + beta * penalty + gamma * t / 25.0
        if objective < best_objective:
            best_objective = objective
            best_result = {
                "optimal_time": t,
                "survival_prob": S_t,
                "coverage": coverage,
                "objective": objective,
                "note": "Risk-aware calculation",
            }
    return best_result
print("\n" + "=" * 60)
print("风险感知的最优NIPT检测时间 (alpha=0.7, beta=0.3, gamma=0.02):")
print("=" * 60)
risk_aware_results = {}
for leaf in np.unique(leaf_id):
    mask = leaf_id == leaf
    group_data = survival_data[mask]
    result = risk_aware_optimal_time(group_data, alpha=0.7, beta=0.3, gamma=0.02)
    risk_aware_results[f"Leaf {leaf}"] = result
    print(f"\nLeaf {leaf}:")
    print(f"  样本量: {len(group_data)}")
    if not np.isnan(result["optimal_time"]):
        print(f"  风险感知最优时间: {result['optimal_time']:.2f} 周")
        print(f"  生存概率: {result['survival_prob']:.3f}")
        print(f"  检测成功率: {result['coverage']:.3f}")
        print(f"  目标函数值: {result['objective']:.4f}")
    else:
        print(f"  状态: {result['note']}")
risk_groups = survival_data["leaf"].values
result = multivariate_logrank_test(time, risk_groups, event)
print("多组 log-rank 检验 p-value:", result.p_value)
def simulate_measurement_error(original_data, error_types):
    """
    模拟多种测量误差对数据的影响
    参数:
    - original_data: 原始数据
    - error_types: 误差类型字典，包含各种误差参数
    """
    simulated_data = original_data.copy()
    n_samples = len(simulated_data)
    concentration_error = error_types.get("concentration_cv", 0.10)  # 默认10%变异系数
    concentration_noise = np.random.normal(1, concentration_error, n_samples)
    gestational_error = error_types.get("gestational_std", 0.7)  # 默认0.7周标准差
    time_noise = np.random.normal(0, gestational_error, n_samples)
    bmi_error = error_types.get("bmi_std", 1.5)  # 默认1.5 kg/m²标准差
    bmi_noise = np.random.normal(0, bmi_error, n_samples)
    simulated_data["Time"] = np.maximum(
        simulated_data["Time"] + time_noise, 8.0  # 确保最小孕周为8周
    )
    simulated_data["BMI"] = np.maximum(
        simulated_data["BMI"] + bmi_noise, 15.0  # 确保BMI合理范围
    )
    detection_success_prob = 1 / (1 + np.exp(-(simulated_data["Time"] - 12) * 0.3))
    detection_success_prob *= concentration_noise  # 浓度误差影响成功率
    random_events = np.random.random(n_samples)
    simulated_data["Event"] = (random_events < detection_success_prob).astype(bool)
    return simulated_data
def monte_carlo_error_analysis(
    original_survival_data, n_simulations=100, error_scenarios=None
):
    """
    蒙特卡罗模拟分析检测误差对最佳时点的影响
    """
    if error_scenarios is None:
        error_scenarios = {
            "low_error": {
                "concentration_cv": 0.05,
                "gestational_std": 0.3,
                "bmi_std": 0.5,
            },
            "medium_error": {
                "concentration_cv": 0.10,
                "gestational_std": 0.7,
                "bmi_std": 1.5,
            },
            "high_error": {
                "concentration_cv": 0.20,
                "gestational_std": 1.2,
                "bmi_std": 2.5,
            },
        }
    results = {}
    for scenario_name, error_params in error_scenarios.items():
        print(f"\n分析误差场景: {scenario_name}")
        print(f"  浓度测量CV: {error_params['concentration_cv']:.1%}")
        print(f"  孕周测量误差: ±{error_params['gestational_std']:.1f}周")
        print(f"  BMI测量误差: ±{error_params['bmi_std']:.1f} kg/m²")
        optimal_times_by_leaf = {leaf: [] for leaf in np.unique(leaf_id)}
        coverage_by_leaf = {leaf: [] for leaf in np.unique(leaf_id)}
        for sim in range(n_simulations):
            if (sim + 1) % 20 == 0:
                print(f"  模拟进度: {sim + 1}/{n_simulations}")
            simulated_data = simulate_measurement_error(
                original_survival_data, error_params
            )
            try:
                y_sim = Surv.from_dataframe("Event", "Time", simulated_data)
                X_sim = (
                    simulated_data[["BMI", "Weight", "Age"]]
                    .to_numpy()
                    .astype(np.float32)
                )
                tree_sim = SurvivalTree(max_leaf_nodes=4, min_samples_leaf=20)
                tree_sim.fit(X_sim, y_sim)
                leaf_id_sim = tree_sim.apply(X_sim)
                simulated_data["leaf"] = leaf_id_sim
                for leaf in np.unique(leaf_id_sim):
                    mask = leaf_id_sim == leaf
                    if mask.sum() < 5:  # 样本量太小跳过
                        continue
                    group_data = simulated_data[mask]
                    result = risk_aware_optimal_time(
                        group_data, alpha=0.7, beta=0.3, gamma=0.02
                    )
                    if not np.isnan(result["optimal_time"]):
                        optimal_times_by_leaf[leaf].append(result["optimal_time"])
                        coverage_by_leaf[leaf].append(result["coverage"])
            except Exception as e:
                continue
        scenario_results = {}
        for leaf in optimal_times_by_leaf:
            if len(optimal_times_by_leaf[leaf]) > 0:
                scenario_results[f"Leaf_{leaf}"] = {
                    "optimal_time_mean": np.mean(optimal_times_by_leaf[leaf]),
                    "optimal_time_std": np.std(optimal_times_by_leaf[leaf]),
                    "optimal_time_ci": np.percentile(
                        optimal_times_by_leaf[leaf], [2.5, 97.5]
                    ),
                    "coverage_mean": np.mean(coverage_by_leaf[leaf]),
                    "coverage_std": np.std(coverage_by_leaf[leaf]),
                    "n_valid_simulations": len(optimal_times_by_leaf[leaf]),
                }
        results[scenario_name] = scenario_results
    return results
print("开始蒙特卡罗模拟分析检测误差对结果的影响...")
print("=" * 60)
error_scenarios = {
    "低误差场景": {
        "concentration_cv": 0.05,  # 5% 浓度测量变异
        "gestational_std": 0.3,  # ±0.3周 孕周测量误差
        "bmi_std": 0.5,  # ±0.5 kg/m² BMI测量误差
    },
    "中等误差场景": {
        "concentration_cv": 0.10,  # 10% 浓度测量变异
        "gestational_std": 0.7,  # ±0.7周 孕周测量误差
        "bmi_std": 1.5,  # ±1.5 kg/m² BMI测量误差
    },
    "高误差场景": {
        "concentration_cv": 0.20,  # 20% 浓度测量变异
        "gestational_std": 1.2,  # ±1.2周 孕周测量误差
        "bmi_std": 2.5,  # ±2.5 kg/m² BMI测量误差
    },
}
error_analysis_results = monte_carlo_error_analysis(
    survival_data, n_simulations=50, error_scenarios=error_scenarios  # 可以根据需要调整
)
print("\n" + "=" * 60)
print("误差分析结果汇总")
print("=" * 60)
for scenario_name, scenario_results in error_analysis_results.items():
    print(f"\n{scenario_name}:")
    print("-" * 40)
    for leaf_name, leaf_results in scenario_results.items():
        print(f"\n  {leaf_name}:")
        print(
            f"    最佳时间: {leaf_results['optimal_time_mean']:.2f} ± {leaf_results['optimal_time_std']:.2f} 周"
        )
        print(
            f"    95% 置信区间: [{leaf_results['optimal_time_ci'][0]:.2f}, {leaf_results['optimal_time_ci'][1]:.2f}] 周"
        )
        print(
            f"    检测成功率: {leaf_results['coverage_mean']:.3f} ± {leaf_results['coverage_std']:.3f}"
        )
        print(f"    有效模拟次数: {leaf_results['n_valid_simulations']}")
original_results = {}
for leaf in np.unique(leaf_id):
    mask = leaf_id == leaf
    group_data = survival_data[mask]
    result = risk_aware_optimal_time(group_data, alpha=0.7, beta=0.3, gamma=0.02)
    if not np.isnan(result["optimal_time"]):
        original_results[f"Leaf_{leaf}"] = {
            "optimal_time": result["optimal_time"],
            "coverage": result["coverage"],
        }
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
leaf_names = []
original_times = []
low_error_times = []
medium_error_times = []
high_error_times = []
low_error_std = []
medium_error_std = []
high_error_std = []
for leaf_name in sorted(original_results.keys()):
    if leaf_name in error_analysis_results["低误差场景"]:
        leaf_names.append(leaf_name.replace("Leaf_", "Group "))
        original_times.append(original_results[leaf_name]["optimal_time"])
        low_error_times.append(
            error_analysis_results["低误差场景"][leaf_name]["optimal_time_mean"]
        )
        low_error_std.append(
            error_analysis_results["低误差场景"][leaf_name]["optimal_time_std"]
        )
        medium_error_times.append(
            error_analysis_results["中等误差场景"][leaf_name]["optimal_time_mean"]
        )
        medium_error_std.append(
            error_analysis_results["中等误差场景"][leaf_name]["optimal_time_std"]
        )
        high_error_times.append(
            error_analysis_results["高误差场景"][leaf_name]["optimal_time_mean"]
        )
        high_error_std.append(
            error_analysis_results["高误差场景"][leaf_name]["optimal_time_std"]
        )
x = np.arange(len(leaf_names))
width = 0.2
ax1.bar(
    x - 1.5 * width,
    original_times,
    width,
    label="No Error (Ideal)",
    color="lightblue",
    alpha=0.8,
)
ax1.bar(
    x - 0.5 * width,
    low_error_times,
    width,
    yerr=low_error_std,
    label="Low Error",
    color="lightgreen",
    alpha=0.8,
    capsize=3,
)
ax1.bar(
    x + 0.5 * width,
    medium_error_times,
    width,
    yerr=medium_error_std,
    label="Medium Error",
    color="orange",
    alpha=0.8,
    capsize=3,
)
ax1.bar(
    x + 1.5 * width,
    high_error_times,
    width,
    yerr=high_error_std,
    label="High Error",
    color="lightcoral",
    alpha=0.8,
    capsize=3,
)
ax1.set_xlabel("Risk Group")
ax1.set_ylabel("Optimal NIPT Time (Weeks)")
ax1.set_title("Impact of Detection Error on Optimal NIPT Time")
ax1.set_xticks(x)
ax1.set_xticklabels(leaf_names)
ax1.legend()
ax1.grid(True, alpha=0.3)
original_coverage = []
low_error_coverage = []
medium_error_coverage = []
high_error_coverage = []
low_coverage_std = []
medium_coverage_std = []
high_coverage_std = []
for leaf_name in sorted(original_results.keys()):
    if leaf_name in error_analysis_results["低误差场景"]:
        original_coverage.append(original_results[leaf_name]["coverage"])
        low_error_coverage.append(
            error_analysis_results["低误差场景"][leaf_name]["coverage_mean"]
        )
        low_coverage_std.append(
            error_analysis_results["低误差场景"][leaf_name]["coverage_std"]
        )
        medium_error_coverage.append(
            error_analysis_results["中等误差场景"][leaf_name]["coverage_mean"]
        )
        medium_coverage_std.append(
            error_analysis_results["中等误差场景"][leaf_name]["coverage_std"]
        )
        high_error_coverage.append(
            error_analysis_results["高误差场景"][leaf_name]["coverage_mean"]
        )
        high_coverage_std.append(
            error_analysis_results["高误差场景"][leaf_name]["coverage_std"]
        )
ax2.bar(
    x - 1.5 * width,
    original_coverage,
    width,
    label="No Error (Ideal)",
    color="lightblue",
    alpha=0.8,
)
ax2.bar(
    x - 0.5 * width,
    low_error_coverage,
    width,
    yerr=low_coverage_std,
    label="Low Error",
    color="lightgreen",
    alpha=0.8,
    capsize=3,
)
ax2.bar(
    x + 0.5 * width,
    medium_error_coverage,
    width,
    yerr=medium_coverage_std,
    label="Medium Error",
    color="orange",
    alpha=0.8,
    capsize=3,
)
ax2.bar(
    x + 1.5 * width,
    high_error_coverage,
    width,
    yerr=high_coverage_std,
    label="High Error",
    color="lightcoral",
    alpha=0.8,
    capsize=3,
)
ax2.set_xlabel("Risk Group")
ax2.set_ylabel("Detection Coverage")
ax2.set_title("Impact of Detection Error on Detection Coverage")
ax2.set_xticks(x)
ax2.set_xticklabels(leaf_names)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("\n" + "=" * 60)
print("检测误差影响分析完成")
print("=" * 60)
def calculate_error_sensitivity():
    """计算检测误差的敏感性指标"""
    sensitivity_results = {}
    for leaf_name in sorted(original_results.keys()):
        if leaf_name in error_analysis_results["低误差场景"]:
            original_time = original_results[leaf_name]["optimal_time"]
            original_coverage = original_results[leaf_name]["coverage"]
            sensitivity_results[leaf_name] = {}
            for scenario in ["低误差场景", "中等误差场景", "高误差场景"]:
                scenario_results = error_analysis_results[scenario][leaf_name]
                time_bias = scenario_results["optimal_time_mean"] - original_time
                time_variability = scenario_results["optimal_time_std"]
                coverage_bias = scenario_results["coverage_mean"] - original_coverage
                coverage_variability = scenario_results["coverage_std"]
                sensitivity_results[leaf_name][scenario] = {
                    "time_bias": time_bias,
                    "time_variability": time_variability,
                    "coverage_bias": coverage_bias,
                    "coverage_variability": coverage_variability,
                    "time_relative_error": abs(time_bias)
                    / original_time
                    * 100,  # 相对误差百分比
                    "coverage_relative_error": abs(coverage_bias)
                    / original_coverage
                    * 100,
                }
    return sensitivity_results
sensitivity_results = calculate_error_sensitivity()
print("\n" + "=" * 60)
print("检测误差敏感性分析")
print("=" * 60)
for leaf_name, leaf_sensitivity in sensitivity_results.items():
    print(f"\n{leaf_name.replace('Leaf_', '风险组 ')}:")
    print("-" * 30)
    for scenario, metrics in leaf_sensitivity.items():
        print(f"\n  {scenario}:")
        print(
            f"    时间偏差: {metrics['time_bias']:+.2f} 周 (相对误差: {metrics['time_relative_error']:.1f}%)"
        )
        print(f"    时间变异性: ±{metrics['time_variability']:.2f} 周")
        print(
            f"    成功率偏差: {metrics['coverage_bias']:+.3f} (相对误差: {metrics['coverage_relative_error']:.1f}%)"
        )
        print(f"    成功率变异性: ±{metrics['coverage_variability']:.3f}")
scenarios_chinese = ["低误差场景", "中等误差场景", "高误差场景"]
scenarios_english = [
    "Low Error Scenario",
    "Medium Error Scenario",
    "High Error Scenario",
]
groups = [
    name.replace("Leaf_", "Group ") for name in sorted(sensitivity_results.keys())
]
time_error_matrix = np.zeros((len(groups), len(scenarios_chinese)))
coverage_error_matrix = np.zeros((len(groups), len(scenarios_chinese)))
for i, leaf_name in enumerate(sorted(sensitivity_results.keys())):
    for j, scenario in enumerate(scenarios_chinese):
        time_error_matrix[i, j] = sensitivity_results[leaf_name][scenario][
            "time_relative_error"
        ]
        coverage_error_matrix[i, j] = sensitivity_results[leaf_name][scenario][
            "coverage_relative_error"
        ]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
im1 = ax1.imshow(time_error_matrix, cmap="Reds", aspect="auto")
ax1.set_xticks(range(len(scenarios_english)))
ax1.set_xticklabels(scenarios_english, rotation=45)
ax1.set_yticks(range(len(groups)))
ax1.set_yticklabels(groups)
ax1.set_title("Relative Error in Optimal NIPT Time (%)")
for i in range(len(groups)):
    for j in range(len(scenarios_english)):
        text = ax1.text(
            j,
            i,
            f"{time_error_matrix[i, j]:.1f}%",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
        )
plt.colorbar(im1, ax=ax1, shrink=0.8)
im2 = ax2.imshow(coverage_error_matrix, cmap="Blues", aspect="auto")
ax2.set_xticks(range(len(scenarios_english)))
ax2.set_xticklabels(scenarios_english, rotation=45)
ax2.set_yticks(range(len(groups)))
ax2.set_yticklabels(groups)
ax2.set_title("Relative Error in Detection Coverage (%)")
for i in range(len(groups)):
    for j in range(len(scenarios_english)):
        text = ax2.text(
            j,
            i,
            f"{coverage_error_matrix[i, j]:.1f}%",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
        )
plt.colorbar(im2, ax=ax2, shrink=0.8)
plt.tight_layout()
plt.show()
print("\n" + "=" * 60)
print("敏感性分析完成")
print("=" * 60)
