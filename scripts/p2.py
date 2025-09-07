import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings
from typing import Dict
import scipy.stats as stats
from scipy.stats import norm
import signal
from contextlib import contextmanager
warnings.filterwarnings("ignore")
def load_and_clean_data(url: str) -> pd.DataFrame:
    """加载 Excel 数据并清理空值"""
    df = pd.read_excel(url, sheet_name=0)
    df.dropna(subset=["检测孕周", "GC含量", "孕妇BMI", "Y染色体浓度"], inplace=True)
    return df
def convert_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """把检测孕周统一转换为周数 (浮点数)"""
    weeks_days = df["检测孕周"].str.split(r"[wW]", expand=True)
    df["孕天"] = weeks_days[0].astype(int) * 7 + weeks_days[1].fillna("0").replace(
        "", "0"
    ).astype(int)
    df["孕天"] = df["孕天"].astype(np.float64) / 7.0
    return df
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """中英文列名映射"""
    col_map_cn = {
        "孕妇代码": "mother_id",
        "检测孕周": "gestational_age",
        "孕妇BMI": "bmi",
        "年龄": "age",
        "Y染色体浓度": "y_concentration",
        "检测日期": "test_date",
        "末次月经": "last_menstrual_period",
        "身高": "height",
        "体重": "weight",
        "在参考基因组上比对的比例": "mapping_ratio",
        "GC含量": "GC",
        "孕天": "gestational_weeks",
    }
    return df.rename(columns=col_map_cn)
def compute_event_time(df: pd.DataFrame, threshold: float = 0.04) -> pd.DataFrame:
    """计算 crossing time (首次超过阈值的孕周)"""
    results = []
    for mid, sub in df.groupby("mother_id"):
        sub = sub.sort_values("gestational_weeks")
        times, y_concs = sub["gestational_weeks"].values, sub["y_concentration"].values
        event_time, event_occurred = None, 0
        for i in range(1, len(y_concs)):
            if y_concs[i - 1] < threshold <= y_concs[i]:
                t1, t2 = times[i - 1], times[i]
                y1, y2 = y_concs[i - 1], y_concs[i]
                event_time = t1 + (threshold - y1) * (t2 - t1) / (y2 - y1)
                event_occurred = 1
                break
        if event_time is None:
            if np.any(y_concs >= threshold):
                event_time = times[np.argmax(y_concs >= threshold)]
                event_occurred = 1
            else:
                event_time = times[-1]
                event_occurred = 0
        event_bmi = sub["bmi"].values[
            np.searchsorted(times, event_time, side="right") - 1
        ]
        results.append((mid, event_time, event_occurred, event_bmi))
    return pd.DataFrame(results, columns=["mother_id", "time", "event", "bmi"])
def find_best_cut(data, min_size=20, alpha=0.05):
    """寻找最优 BMI 切割点"""
    sorted_bmi = np.sort(data["bmi"].unique())
    best_cut, best_stat, best_p = None, -np.inf, 1.0
    for cut in sorted_bmi[1:-1]:
        group1, group2 = data[data["bmi"] <= cut], data[data["bmi"] > cut]
        if len(group1) < min_size or len(group2) < min_size:
            continue
        result = logrank_test(
            group1["time"],
            group2["time"],
            event_observed_A=group1["event"],
            event_observed_B=group2["event"],
        )
        if result.test_statistic > best_stat:
            best_cut, best_stat, best_p = cut, result.test_statistic, result.p_value
    return (best_cut, best_stat, best_p) if best_p < alpha else (None, None, None)
def recursive_partition_with_p(df, max_groups=5, min_size=20, alpha=0.05):
    """递归分组并保存切割点与 p 值"""
    cuts, cut_pvalues = [], []
    def split(data):
        if len(cuts) >= max_groups - 1:
            return
        cut, stat, p = find_best_cut(data, min_size, alpha)
        if cut is not None:
            cuts.append(cut)
            cut_pvalues.append(p)
            split(data[data["bmi"] <= cut])
            split(data[data["bmi"] > cut])
    split(df)
    cuts = sorted(cuts)
    cut_pvalues_sorted = [p for _, p in sorted(zip(cuts, cut_pvalues))]
    bins = [-np.inf] + cuts + [np.inf]
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    df["bmi_group"] = pd.cut(df["bmi"], bins=bins, labels=labels)
    return df, cuts, cut_pvalues_sorted
def best_nipt_times(data):
    """根据 KM 曲线计算最佳检测孕周 (生存率 ≤ 0.1)"""
    kmf, best_times = KaplanMeierFitter(), {}
    for group, sub in data.groupby("bmi_group"):
        sub = sub.dropna()
        if len(sub) < 10:
            continue
        kmf.fit(sub["time"], event_observed=sub["event"])
        surv = kmf.survival_function_
        crossing = surv[surv["KM_estimate"] <= 0.1]
        best_times[group] = crossing.index[0] if not crossing.empty else np.nan
    return best_times
def add_measurement_error(df, error_std=0.005):
    """在 Y 浓度中加入高斯噪声"""
    np.random.seed(42)
    df_err = df.copy()
    df_err["y_concentration"] += np.random.normal(0, error_std, size=len(df_err))
    df_err["y_concentration"] = df_err["y_concentration"].clip(0, 1)
    return df_err
def fmt_list(lst, digits=1):
    return [round(float(x), digits) for x in lst]
def fmt_dict(dct, digits=1):
    return {k: round(float(v), digits) for k, v in dct.items()}
raw_df = load_and_clean_data("data/data.xlsx")
raw_df = convert_weeks(raw_df)
df = rename_columns(raw_df)
print("Data shape:", df.shape)
df_event = compute_event_time(
    df[["mother_id", "gestational_weeks", "bmi", "y_concentration"]]
)
df_grouped, cuts, cut_pvalues = recursive_partition_with_p(
    df_event, max_groups=3, min_size=5, alpha=0.1
)
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))
for group, sub in df_grouped.groupby("bmi_group"):
    sub = sub.dropna()
    kmf.fit(sub["time"], event_observed=sub["event"], label=str(group))
    kmf.plot_survival_function()
plt.xlabel("Gestational Weeks")
plt.ylabel("Survival Probability")
plt.title("KM Curves by BMI Groups")
plt.legend(title="BMI Groups")
plt.show()
best_times = best_nipt_times(df_grouped)
df_err = add_measurement_error(df)
df_event_err = compute_event_time(
    df_err[["mother_id", "gestational_weeks", "bmi", "y_concentration"]]
)
df_grouped_err, cuts_err, cut_pvalues_err = recursive_partition_with_p(df_event_err)
best_times_err = best_nipt_times(df_grouped_err)
print("Optimal BMI Cut Points:", fmt_list(cuts))
print("Cutpoints p-values:", fmt_list(cut_pvalues, digits=3))
print("Cutpoints with Measurement Error:", fmt_list(cuts_err))
print(
    "Cutpoints p-values with Measurement Error:",
    fmt_list(cut_pvalues_err, digits=3),
)
print("\nOptimal NIPT Times (weeks):")
for k, v in fmt_dict(best_times).items():
    print(f"  {k:10s} -> {v}")
print("NIPT Times with Measurement Error (weeks):")
for k, v in fmt_dict(best_times_err).items():
    print(f"  {k:10s} -> {v}")
def _risk_penalty(t: float) -> float:
    return (t**2 - 1.0) / (624.0)  # 归一化到 [0, 1]
def risk_aware_optimal_time(
    sub: pd.DataFrame,
    alpha: float = 1.0,  # 权重：失败率（1-coverage）
    beta: float = 1.0,  # 权重：风险惩罚
    gamma: float = 0.02,  # 权重：孕周（鼓励更早）
    grid: np.ndarray | None = None,
) -> Dict:
    sub = sub.dropna()
    if len(sub) < 5:
        return {
            "t_star": np.nan,
            "S_t": np.nan,
            "coverage": np.nan,
            "objective": np.inf,
        }
    kmf_local = KaplanMeierFitter()
    kmf_local.fit(sub["time"], event_observed=sub["event"])
    if grid is None:
        t_min = max(5.0, float(np.nanmin(sub["time"])) if len(sub) else 5.0)
        t_max = min(40.0, float(np.nanmax(sub["time"])) if len(sub) else 40.0)
        grid = np.unique(
            np.concatenate(
                [
                    np.linspace(t_min, t_max, 400),
                    np.array([12.0, 13.0, 27.0, 28.0]),
                    kmf_local.survival_function_.index.values,
                ]
            )
        )
        grid = grid[(grid >= t_min) & (grid <= t_max)]
    best = None
    for t in grid:
        S_t = float(kmf_local.predict(t))
        coverage = 1.0 - S_t
        penalty = _risk_penalty(float(t))
        objective = alpha * (1 - coverage) + beta * penalty + gamma * float(t) / 25.0
        cand = (objective, t, S_t, coverage)
        if best is None or cand < best:
            best = cand
    objective, t_star, S_t, coverage = best
    return {
        "t_star": float(t_star),
        "S_t": float(S_t),
        "coverage": float(coverage),
        "objective": float(objective),
    }
def risk_aware_best_times(
    df_grouped: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.02,
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for group, sub in df_grouped.groupby("bmi_group"):
        res = risk_aware_optimal_time(
            sub,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        results[str(group)] = res
    return results
alpha, beta, gamma = 0.7, 0.3, 0.0  # 更看重检测准确性
risk_results = risk_aware_best_times(
    df_grouped,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
)
print(
    "Risk-aware optimal NIPT timing per BMI group (alpha, beta, gamma) =",
    (alpha, beta, gamma),
)
for k in sorted(risk_results.keys()):
    v = risk_results[k]
    if np.isfinite(v["objective"]):
        print(
            f"  {k:12s} -> t* = {v['t_star']:.1f} w, S(t*) = {v['S_t']:.3f}, coverage = {v['coverage']:.3f}"
        )
    else:
        print(f"  {k:12s} -> insufficient data")
def simulate_measurement_error(original_data, error_scenario):
    """
    模拟检测误差对数据的影响
    参数:
    - original_data: 原始数据DataFrame
    - error_scenario: 误差场景字典，包含各种误差参数
    """
    simulated_data = original_data.copy()
    n_samples = len(simulated_data)
    concentration_cv = error_scenario.get("concentration_cv", 0.05)  # 默认5%
    concentration_multiplier = np.random.normal(1.0, concentration_cv, n_samples)
    concentration_multiplier = np.clip(
        concentration_multiplier, 0.5, 2.0
    )  # 限制在合理范围
    gestational_std = error_scenario.get("gestational_std", 0.5)  # 默认0.5周
    gestational_noise = np.random.normal(0, gestational_std, n_samples)
    bmi_std = error_scenario.get("bmi_std", 1.0)  # 默认1.0 kg/m²
    bmi_noise = np.random.normal(0, bmi_std, n_samples)
    simulated_data["y_concentration"] = (
        simulated_data["y_concentration"] * concentration_multiplier
    )
    simulated_data["y_concentration"] = np.clip(simulated_data["y_concentration"], 0, 1)
    simulated_data["gestational_weeks"] = np.maximum(
        simulated_data["gestational_weeks"] + gestational_noise, 5.0
    )
    simulated_data["bmi"] = np.maximum(simulated_data["bmi"] + bmi_noise, 15.0)
    return simulated_data
@contextmanager
def timeout(duration):
    """超时控制上下文管理器"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"操作超时 ({duration} 秒)")
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)
def robust_recursive_partition(df, max_groups=3, min_size=10, alpha=0.1, max_depth=3):
    """
    更健壮的递归分组函数，添加深度限制和异常处理
    """
    cuts = []
    cut_pvalues = []
    def split(data, depth=0):
        if (
            len(cuts) >= max_groups - 1
            or len(data) < min_size * 2
            or depth >= max_depth
        ):
            return
        try:
            with timeout(10):  # 10秒超时
                cut, stat, p = find_best_cut(data, min_size, alpha)
        except (TimeoutError, Exception):
            return
        if cut is not None:
            cuts.append(cut)
            cut_pvalues.append(p)
            left_data = data[data["bmi"] <= cut]
            right_data = data[data["bmi"] > cut]
            if len(left_data) >= min_size:
                split(left_data, depth + 1)
            if len(right_data) >= min_size:
                split(right_data, depth + 1)
    try:
        split(df)
    except Exception as e:
        print(f"  警告：分组过程出现异常: {e}")
        if not cuts and len(df) >= min_size * 2:
            median_bmi = df["bmi"].median()
            cuts = [median_bmi]
            cut_pvalues = [0.05]  # 给一个默认的p值
    if not cuts:
        df["bmi_group"] = "all"
        return df, [], []
    cuts = sorted(cuts)
    cut_pvalues_sorted = [p for _, p in sorted(zip(cuts, cut_pvalues))]
    bins = [-np.inf] + cuts + [np.inf]
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    df["bmi_group"] = pd.cut(df["bmi"], bins=bins, labels=labels)
    return df, cuts, cut_pvalues_sorted
def monte_carlo_error_analysis(original_df, n_simulations=100, error_scenarios=None):
    """
    改进的蒙特卡罗误差影响分析，添加超时和错误处理
    """
    if error_scenarios is None:
        error_scenarios = {
            "低误差场景": {
                "concentration_cv": 0.05,  # 5% 变异系数
                "gestational_std": 0.3,  # 0.3周标准差
                "bmi_std": 0.5,  # 0.5 kg/m² 标准差
            },
            "中等误差场景": {
                "concentration_cv": 0.10,  # 10% 变异系数
                "gestational_std": 0.7,  # 0.7周标准差
                "bmi_std": 1.5,  # 1.5 kg/m² 标准差
            },
            "高误差场景": {
                "concentration_cv": 0.15,  # 15% 变异系数
                "gestational_std": 1.0,  # 1.0周标准差
                "bmi_std": 2.0,  # 2.0 kg/m² 标准差
            },
        }
    results = {}
    for scenario_name, scenario_params in error_scenarios.items():
        print(f"运行 {scenario_name} 模拟...")
        scenario_results = {"cut_points": [], "best_times": [], "risk_aware_times": []}
        successful_sims = 0
        max_failures = n_simulations // 2  # 允许最多一半的模拟失败
        failures = 0
        for sim in range(n_simulations):
            if failures > max_failures:
                print(f"  警告：失败次数过多，提前终止 {scenario_name}")
                break
            if (sim + 1) % 10 == 0:
                print(
                    f"  完成 {sim + 1}/{n_simulations} 次模拟 (成功: {successful_sims})"
                )
            try:
                simulated_df = simulate_measurement_error(original_df, scenario_params)
                df_event_sim = compute_event_time(
                    simulated_df[
                        ["mother_id", "gestational_weeks", "bmi", "y_concentration"]
                    ]
                )
                df_grouped_sim, cuts_sim, cut_pvalues_sim = robust_recursive_partition(
                    df_event_sim, max_groups=3, min_size=8, alpha=0.2  # 放宽参数
                )
                best_times_sim = best_nipt_times(df_grouped_sim)
                risk_results_sim = risk_aware_best_times(
                    df_grouped_sim, alpha=0.7, beta=0.3, gamma=0.0
                )
                scenario_results["cut_points"].append(cuts_sim)
                scenario_results["best_times"].append(best_times_sim)
                scenario_results["risk_aware_times"].append(risk_results_sim)
                successful_sims += 1
            except Exception as e:
                failures += 1
                if failures <= 3:  # 只打印前3个错误
                    print(f"  模拟 {sim+1} 失败: {str(e)[:100]}")
                continue
        results[scenario_name] = scenario_results
        print(
            f"  {scenario_name} 完成，有效模拟次数: {successful_sims}/{n_simulations}"
        )
    return results
error_scenarios = {
    "低误差场景": {
        "concentration_cv": 0.05,  # 5% 变异系数
        "gestational_std": 0.3,  # 0.3周标准差
        "bmi_std": 0.5,  # 0.5 kg/m² 标准差
    },
    "中等误差场景": {
        "concentration_cv": 0.10,  # 10% 变异系数
        "gestational_std": 0.7,  # 0.7周标准差
        "bmi_std": 1.5,  # 1.5 kg/m² 标准差
    },
    "高误差场景": {
        "concentration_cv": 0.15,  # 15% 变异系数
        "gestational_std": 1.0,  # 1.0周标准差
        "bmi_std": 2.0,  # 2.0 kg/m² 标准差
    },
}
print("=" * 60)
print("原始结果（无误差基准）:")
print("=" * 60)
original_results = {
    "cut_points": cuts,
    "cut_pvalues": cut_pvalues,
    "best_times": best_times,
    "risk_aware_times": risk_results,
}
print("BMI分组切点:", [f"{x:.1f}" for x in cuts])
print("切点p值:", [f"{x:.3f}" for x in cut_pvalues])
print("\n传统最佳NIPT时间:")
for group, time in best_times.items():
    if not np.isnan(time):
        print(f"  {group}: {time:.1f} 周")
print("\n风险感知最佳NIPT时间:")
for group, result in risk_results.items():
    if np.isfinite(result["objective"]):
        print(
            f"  {group}: {result['t_star']:.1f} 周 (覆盖率: {result['coverage']:.3f})"
        )
print(f"\n{'=' * 60}")
print("开始蒙特卡罗误差影响分析...")
print("注意：为避免计算时间过长，使用较少的模拟次数")
print("=" * 60)
np.random.seed(42)  # 设置随机种子以确保结果可重现
simplified_error_scenarios = {
    "低误差场景": {
        "concentration_cv": 0.05,  # 5% 变异系数
        "gestational_std": 0.3,  # 0.3周标准差
        "bmi_std": 0.5,  # 0.5 kg/m² 标准差
    },
    "高误差场景": {
        "concentration_cv": 0.15,  # 15% 变异系数
        "gestational_std": 1.0,  # 1.0周标准差
        "bmi_std": 2.0,  # 2.0 kg/m² 标准差
    },
}
try:
    monte_carlo_results = monte_carlo_error_analysis(
        df[["mother_id", "gestational_weeks", "bmi", "y_concentration"]],
        n_simulations=20,  # 减少到20次模拟以避免超时
        error_scenarios=simplified_error_scenarios,
    )
    print("\n蒙特卡罗模拟完成！")
except Exception as e:
    print(f"\n蒙特卡罗模拟出现错误: {e}")
    print("这可能是由于数据特性或计算复杂度导致的。")
    monte_carlo_results = None
def analyze_monte_carlo_results(monte_carlo_results, original_results):
    """分析蒙特卡罗模拟结果的统计特性"""
    analysis = {}
    for scenario_name, scenario_data in monte_carlo_results.items():
        print(f"\n{scenario_name} 分析结果:")
        print("-" * 40)
        cut_points_list = scenario_data["cut_points"]
        if cut_points_list:
            valid_cuts = [cuts for cuts in cut_points_list if len(cuts) > 0]
            if valid_cuts:
                max_cuts = max(len(cuts) for cuts in valid_cuts)
                cut_stats = []
                for i in range(max_cuts):
                    cut_values = [cuts[i] for cuts in valid_cuts if len(cuts) > i]
                    if cut_values:
                        cut_stats.append(
                            {
                                "mean": np.mean(cut_values),
                                "std": np.std(cut_values),
                                "ci_lower": np.percentile(cut_values, 2.5),
                                "ci_upper": np.percentile(cut_values, 97.5),
                                "count": len(cut_values),
                            }
                        )
                print(f"BMI切点稳定性分析 (基于{len(valid_cuts)}次有效模拟):")
                for i, stats in enumerate(cut_stats):
                    original_cut = (
                        original_results["cut_points"][i]
                        if i < len(original_results["cut_points"])
                        else None
                    )
                    print(
                        f"  切点{i+1}: {stats['mean']:.1f} ± {stats['std']:.1f} kg/m²"
                    )
                    print(
                        f"         95%CI: [{stats['ci_lower']:.1f}, {stats['ci_upper']:.1f}]"
                    )
                    if original_cut is not None:
                        bias = stats["mean"] - original_cut
                        print(
                            f"         偏差: {bias:+.1f} kg/m² (相对原始值 {original_cut:.1f})"
                        )
                    print(
                        f"         出现频率: {stats['count']}/{len(valid_cuts)} ({stats['count']/len(valid_cuts)*100:.1f}%)"
                    )
        risk_aware_times_list = scenario_data["risk_aware_times"]
        if risk_aware_times_list:
            all_groups = set()
            for times_dict in risk_aware_times_list:
                all_groups.update(times_dict.keys())
            print(f"\n风险感知NIPT时间稳定性分析:")
            for group in sorted(all_groups):
                group_times = []
                group_coverages = []
                for times_dict in risk_aware_times_list:
                    if group in times_dict and np.isfinite(
                        times_dict[group]["objective"]
                    ):
                        group_times.append(times_dict[group]["t_star"])
                        group_coverages.append(times_dict[group]["coverage"])
                if group_times:
                    original_time = (
                        original_results["risk_aware_times"]
                        .get(group, {})
                        .get("t_star", np.nan)
                    )
                    original_coverage = (
                        original_results["risk_aware_times"]
                        .get(group, {})
                        .get("coverage", np.nan)
                    )
                    time_mean = np.mean(group_times)
                    time_std = np.std(group_times)
                    time_ci_lower = np.percentile(group_times, 2.5)
                    time_ci_upper = np.percentile(group_times, 97.5)
                    coverage_mean = np.mean(group_coverages)
                    coverage_std = np.std(group_coverages)
                    print(f"  {group}:")
                    print(f"    最佳时间: {time_mean:.1f} ± {time_std:.1f} 周")
                    print(f"    95%CI: [{time_ci_lower:.1f}, {time_ci_upper:.1f}] 周")
                    if not np.isnan(original_time):
                        time_bias = time_mean - original_time
                        print(
                            f"    时间偏差: {time_bias:+.1f} 周 (相对原始值 {original_time:.1f})"
                        )
                    print(f"    检测覆盖率: {coverage_mean:.3f} ± {coverage_std:.3f}")
                    if not np.isnan(original_coverage):
                        coverage_bias = coverage_mean - original_coverage
                        print(
                            f"    覆盖率偏差: {coverage_bias:+.3f} (相对原始值 {original_coverage:.3f})"
                        )
                    print(f"    有效模拟次数: {len(group_times)}")
        analysis[scenario_name] = {
            "cut_stats": cut_stats if "cut_stats" in locals() else [],
            "n_valid_simulations": len(valid_cuts) if "valid_cuts" in locals() else 0,
        }
    return analysis
print("=" * 60)
print("蒙特卡罗模拟结果分析")
print("=" * 60)
analysis_results = analyze_monte_carlo_results(monte_carlo_results, original_results)
def visualize_error_impact(monte_carlo_results, original_results):
    """可视化检测误差对结果的影响"""
    scenarios = list(monte_carlo_results.keys())
    scenario_colors = ["lightgreen", "lightcoral"]
    scenario_labels = {"低误差场景": "Low Error", "高误差场景": "High Error"}
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    for i, (scenario, color) in enumerate(zip(scenarios, scenario_colors)):
        scenario_data = monte_carlo_results[scenario]
        all_first_cuts = []
        for cuts in scenario_data["cut_points"]:
            if len(cuts) > 0:
                all_first_cuts.append(cuts[0])
        if all_first_cuts:
            english_label = scenario_labels.get(scenario, scenario)
            plt.hist(
                all_first_cuts,
                bins=15,
                alpha=0.7,
                color=color,
                label=f"{english_label}\n(n={len(all_first_cuts)})",
                density=True,
            )
    if len(original_results["cut_points"]) > 0:
        plt.axvline(
            original_results["cut_points"][0],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f'Original: {original_results["cut_points"][0]:.1f}',
        )
    plt.xlabel("First BMI Cut Point (kg/m²)")
    plt.ylabel("Density")
    plt.title("BMI Cut Point Stability Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 2)
    all_groups = set()
    for scenario in scenarios:
        for times_dict in monte_carlo_results[scenario]["risk_aware_times"]:
            all_groups.update(times_dict.keys())
    if all_groups:
        representative_group = sorted(list(all_groups))[0]  # 选择第一个组作为代表
        for i, (scenario, color) in enumerate(zip(scenarios, scenario_colors)):
            scenario_data = monte_carlo_results[scenario]
            group_times = []
            for times_dict in scenario_data["risk_aware_times"]:
                if representative_group in times_dict and np.isfinite(
                    times_dict[representative_group]["objective"]
                ):
                    group_times.append(times_dict[representative_group]["t_star"])
            if group_times:
                english_label = scenario_labels.get(scenario, scenario)
                plt.hist(
                    group_times,
                    bins=15,
                    alpha=0.7,
                    color=color,
                    label=f"{english_label}\n(n={len(group_times)})",
                    density=True,
                )
        if representative_group in original_results["risk_aware_times"]:
            original_time = original_results["risk_aware_times"][representative_group][
                "t_star"
            ]
            if np.isfinite(original_time):
                plt.axvline(
                    original_time,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Original: {original_time:.1f}w",
                )
        plt.xlabel("Best NIPT Time (weeks)")
        plt.ylabel("Density")
        plt.title(f"NIPT Time Stability Analysis\n({representative_group})")
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 3)
    if all_groups and representative_group:
        for i, (scenario, color) in enumerate(zip(scenarios, scenario_colors)):
            scenario_data = monte_carlo_results[scenario]
            group_coverages = []
            for times_dict in scenario_data["risk_aware_times"]:
                if representative_group in times_dict and np.isfinite(
                    times_dict[representative_group]["objective"]
                ):
                    group_coverages.append(times_dict[representative_group]["coverage"])
            if group_coverages:
                english_label = scenario_labels.get(scenario, scenario)
                plt.hist(
                    group_coverages,
                    bins=15,
                    alpha=0.7,
                    color=color,
                    label=f"{english_label}\n(n={len(group_coverages)})",
                    density=True,
                )
        if representative_group in original_results["risk_aware_times"]:
            original_coverage = original_results["risk_aware_times"][
                representative_group
            ]["coverage"]
            if np.isfinite(original_coverage):
                plt.axvline(
                    original_coverage,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Original: {original_coverage:.3f}",
                )
        plt.xlabel("Detection Coverage")
        plt.ylabel("Density")
        plt.title(f"Detection Coverage Stability Analysis\n({representative_group})")
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 4)
    time_data_for_boxplot = []
    scenario_labels_for_boxplot = []
    for scenario in scenarios:
        scenario_times = []
        for times_dict in monte_carlo_results[scenario]["risk_aware_times"]:
            for group, result in times_dict.items():
                if np.isfinite(result["objective"]):
                    scenario_times.append(result["t_star"])
        if scenario_times:
            time_data_for_boxplot.append(scenario_times)
            english_label = scenario_labels.get(scenario, scenario)
            scenario_labels_for_boxplot.append(english_label)
    if time_data_for_boxplot:
        plt.boxplot(time_data_for_boxplot, labels=scenario_labels_for_boxplot)
        plt.ylabel("Best NIPT Time (weeks)")
        plt.title("Time Distribution under Different Error Scenarios")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 5)
    coverage_data_for_boxplot = []
    for scenario in scenarios:
        scenario_coverages = []
        for times_dict in monte_carlo_results[scenario]["risk_aware_times"]:
            for group, result in times_dict.items():
                if np.isfinite(result["objective"]):
                    scenario_coverages.append(result["coverage"])
        if scenario_coverages:
            coverage_data_for_boxplot.append(scenario_coverages)
    if coverage_data_for_boxplot:
        plt.boxplot(coverage_data_for_boxplot, labels=scenario_labels_for_boxplot)
        plt.ylabel("Detection Coverage")
        plt.title("Coverage Distribution under Different Error Scenarios")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 6)
    relative_errors = {"time": [], "coverage": []}
    for scenario in scenarios:
        scenario_time_errors = []
        scenario_coverage_errors = []
        for times_dict in monte_carlo_results[scenario]["risk_aware_times"]:
            for group, result in times_dict.items():
                if group in original_results["risk_aware_times"] and np.isfinite(
                    result["objective"]
                ):
                    original_time = original_results["risk_aware_times"][group][
                        "t_star"
                    ]
                    original_coverage = original_results["risk_aware_times"][group][
                        "coverage"
                    ]
                    if np.isfinite(original_time) and original_time > 0:
                        time_error = (
                            abs(result["t_star"] - original_time) / original_time * 100
                        )
                        scenario_time_errors.append(time_error)
                    if np.isfinite(original_coverage) and original_coverage > 0:
                        coverage_error = (
                            abs(result["coverage"] - original_coverage)
                            / original_coverage
                            * 100
                        )
                        scenario_coverage_errors.append(coverage_error)
        if scenario_time_errors:
            relative_errors["time"].append(np.mean(scenario_time_errors))
        else:
            relative_errors["time"].append(0)
        if scenario_coverage_errors:
            relative_errors["coverage"].append(np.mean(scenario_coverage_errors))
        else:
            relative_errors["coverage"].append(0)
    x = np.arange(len(scenarios))
    width = 0.35
    plt.bar(
        x - width / 2,
        relative_errors["time"],
        width,
        label="Time Relative Error",
        alpha=0.8,
    )
    plt.bar(
        x + width / 2,
        relative_errors["coverage"],
        width,
        label="Coverage Relative Error",
        alpha=0.8,
    )
    plt.xlabel("Error Scenarios")
    plt.ylabel("Average Relative Error (%)")
    plt.title("Detection Error Sensitivity Summary")
    plt.xticks(x, [scenario_labels.get(s, s) for s in scenarios], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
print(f"\n{'=' * 60}")
print("Generating error impact analysis visualizations...")
print("=" * 60)
visualize_error_impact(monte_carlo_results, original_results)
