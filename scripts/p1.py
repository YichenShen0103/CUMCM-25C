import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import f
from scipy.stats import pearsonr, spearmanr
import warnings
from pygam import LinearGAM, s
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
warnings.filterwarnings("ignore")
url = "https://raw.githubusercontent.com/YichenShen0103/CUMCM-25C/main/data/data.xlsx"
data = pd.read_excel(url, sheet_name=0)
data.dropna(subset=["检测孕周", "GC含量", "孕妇BMI", "Y染色体浓度"], inplace=True)
weeks_days = data["检测孕周"].str.split(r"[wW]", expand=True)
data["孕天"] = weeks_days[0].astype(int) * 7 + weeks_days[1].fillna("0").replace(
    "", "0"
).astype(int)
data["检测日期"] = pd.to_datetime(data["检测日期"], format="%Y%m%d")
data["末次月经"] = pd.to_datetime(data["末次月经"], format="%Y-%m-%d")
data["delta_days"] = (data["检测日期"] - data["末次月经"]).dt.days - data["孕天"]
data = data[abs(data["delta_days"]) <= 0]
data.drop(columns=["delta_days", "生产次数"], inplace=True)
day_mean = data["孕天"].mean()
day_std = data["孕天"].std()
data = data[
    (data["孕天"] <= day_mean + 3 * day_std) & (data["孕天"] >= day_mean - 3 * day_std)
]
num_cols = data.select_dtypes(include=["float64", "int64"]).columns
fig, axes = plt.subplots(nrows=len(num_cols), ncols=1, figsize=(8, 3 * len(num_cols)))
for i, col in enumerate(num_cols):
    sns.boxplot(x=data[col], ax=axes[i])
    axes[i].set_title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()
for col in num_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data[col] >= Q1 - 3 * IQR) & (data[col] <= Q3 + 3 * IQR)]
mod_df = data[
    [
        "孕天",
        "Y染色体浓度",
        "X染色体浓度",
        "13号染色体的GC含量",
        "18号染色体的GC含量",
        "21号染色体的GC含量",
        "GC含量",
        "孕妇BMI",
        "孕妇代码",
    ]
]
data = data[
    [
        "孕天",
        "Y染色体浓度",
        "X染色体浓度",
        "13号染色体的GC含量",
        "18号染色体的GC含量",
        "21号染色体的GC含量",
        "GC含量",
        "孕妇BMI",
    ]
]
dictionary = {
    "孕天": "GA (days)",
    "孕妇BMI": "BMI",
    "Y染色体浓度": "Y CC",
    "X染色体浓度": "X CC",
    "13号染色体的GC含量": "13 GC",
    "18号染色体的GC含量": "18 GC",
    "21号染色体的GC含量": "21 GC",
    "GC含量": "GC",
}
corr = data.corr()
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.imshow(corr, cmap="seismic", vmin=-1, vmax=1)
fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(np.arange(len(corr)))
ax.set_yticks(np.arange(len(corr)))
ax.set_xticklabels([dictionary[col] for col in corr.columns], rotation=45, ha="right")
ax.set_yticklabels([dictionary[col] for col in corr.columns])
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
ax.set_title("Heatmap of Correlation Matrix")
plt.tight_layout()
plt.show()
plt.scatter(data["孕天"], data["Y染色体浓度"])
plt.xlabel("Pregnancy days")
plt.ylabel("Y chromosome concentration")
plt.title("Pregnancy days vs Y chromosome concentration")
plt.show()
plt.scatter(data["孕妇BMI"], data["Y染色体浓度"])
plt.xlabel("Pregnant BMI")
plt.ylabel("Y chromosome concentration")
plt.title("Pregnant BMI vs Y chromosome concentration")
plt.show()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    data["孕妇BMI"],
    data["孕天"],
    data["Y染色体浓度"],
    c=data["Y染色体浓度"],
    cmap="viridis",
    s=50,
)
ax.set_xlabel("X1 (Pregnant BMI)")
ax.set_ylabel("X2 (Pregnancy days)")
ax.set_zlabel("Y (Y chromosome concentration)")
plt.colorbar(sc, label="Y chromosome concentration")
plt.show()
Y = data["Y染色体浓度"]
X1 = data["孕妇BMI"]
X2 = np.log(data["孕天"])
print("X1: BMI, X2: log(孕周), Y: Y染色体浓度")
corr, p_value = pearsonr(X1, Y)
corr_spearman, p_value_spearman = spearmanr(X1, Y)
print("X1, Y person r=", corr, " p=", p_value)
print("X1, Y spearman r=", corr_spearman, " p=", p_value_spearman)
corr, p_value = pearsonr(X2, Y)
corr_spearman, p_value_spearman = spearmanr(X2, Y)
print("X2, Y person r=", corr, " p=", p_value)
print("X2, Y spearman r=", corr_spearman, " p=", p_value_spearman)
X = data[["孕妇BMI", "孕天"]].values
y = data["Y染色体浓度"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
gam = LinearGAM(s(0) + s(1)).fit(X_train, y_train)
y_pred_train = gam.predict(X_train)
y_pred_test = gam.predict(X_test)
print("=== GAM模型效果评估 ===")
print(f"训练集 R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"测试集 R²: {r2_score(y_test, y_pred_test):.4f}")
print(f"训练集 RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"测试集 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print(f"训练集 MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"测试集 MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"\n=== GAM模型统计指标 ===")
print("可用的统计属性:", list(gam.statistics_.keys()))
stats = gam.statistics_
if "AIC" in stats:
    print(f"AIC: {stats['AIC']:.4f}")
if "GCV" in stats:
    print(f"GCV: {stats['GCV']:.6f}")
if "pseudo_r2" in stats:
    print(f"Pseudo R²: {stats['pseudo_r2']}")
if "edof" in stats:
    print(f"有效自由度: {stats['edof']:.2f}")
explained_variance_ratio = r2_score(y_train, y_pred_train)
print(f"解释方差比例: {explained_variance_ratio:.4f}")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
feature_names = ["孕妇BMI", "孕天"]
for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue
    XX = gam.generate_X_grid(term=i)
    pdep = gam.partial_dependence(term=i, X=XX)
    axes[i].plot(XX[:, i], pdep, "b-", linewidth=2)
    axes[i].set_xlabel(feature_names[i])
    axes[i].set_ylabel(f"Partial dependence function (f{i+1})")
    axes[i].set_title(f"{feature_names[i]}'s non-linear effect")
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
axes[0, 0].set_xlabel("True value")
axes[0, 0].set_ylabel("Predicted value")
axes[0, 0].set_title("Predicted vs True values")
axes[0, 0].grid(True, alpha=0.3)
r2_test = r2_score(y_test, y_pred_test)
axes[0, 0].text(
    0.05,
    0.95,
    f"R² = {r2_test:.4f}",
    transform=axes[0, 0].transAxes,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)
residuals = y_test - y_pred_test
axes[0, 1].scatter(y_pred_test, residuals, alpha=0.6)
axes[0, 1].axhline(y=0, color="r", linestyle="--")
axes[0, 1].set_xlabel("predicted values")
axes[0, 1].set_ylabel("residuals")
axes[0, 1].set_title("residuals vs Predicted values")
axes[0, 1].grid(True, alpha=0.3)
axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor="black")
axes[1, 0].set_xlabel("residuals")
axes[1, 0].set_ylabel("frequency")
axes[1, 0].set_title("Residuals Distribution")
axes[1, 0].grid(True, alpha=0.3)
from scipy.stats import shapiro
shapiro_stat, shapiro_p = shapiro(residuals)
axes[1, 0].text(
    0.05,
    0.95,
    f"Shapiro-Wilk test\np = {shapiro_p:.4f}",
    transform=axes[1, 0].transAxes,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Residual Q-Q Graph")
axes[1, 1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print(f"\n=== 模型显著性检验 ===")
n = len(y_train)
k = 2  # 两个变量
ss_res = np.sum((y_train - y_pred_train) ** 2)
ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
ss_reg = ss_tot - ss_res
mse_reg = ss_reg / k
mse_res = ss_res / (n - k - 1)
f_stat = mse_reg / mse_res
p_value = 1 - f.cdf(f_stat, k, n - k - 1)
print(f"F统计量: {f_stat:.4f}")
print(f"p值: {p_value:.4e}")
print(f"模型{'显著' if p_value < 0.05 else '不显著'} (α=0.05)")
print(f"\n=== 残差正态性检验 ===")
print(f"Shapiro-Wilk统计量: {shapiro_stat:.4f}")
print(f"Shapiro-Wilk p值: {shapiro_p:.4f}")
print(f"残差{'服从' if shapiro_p > 0.05 else '不服从'}正态分布 (α=0.05)")
print(f"\n=== 模型总结 ===")
print(
    f"• 模型在训练集上的R²为{r2_score(y_train, y_pred_train):.4f}，在测试集上的R²为{r2_score(y_test, y_pred_test):.4f}"
)
print(f"• 测试集RMSE为{np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print(f"• 模型整体{'显著' if p_value < 0.05 else '不显著'}")
print(f"• 残差{'服从' if shapiro_p > 0.05 else '不服从'}正态分布假设")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
y_pred = gam.predict(X)
residuals = y - y_pred
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    X[:, 0],  # 孕妇BMI
    X[:, 1],  # 孕天
    residuals,  # 残差
    c=residuals,
    cmap="coolwarm",
    s=50,
    alpha=0.8,
)
ax.set_xlabel("Preg BMI")
ax.set_ylabel("Preg Days")
ax.set_zlabel("Residuals (True - Predicted)")
ax.set_title("3D Scatter of Residuals")
plt.colorbar(sc, ax=ax, shrink=0.6, label="Residual")
plt.show()
data["log孕天"] = np.log(data["孕天"])
X = data[["孕妇BMI", "log孕天"]]
y = data["Y染色体浓度"]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
plt.scatter(data["孕妇BMI"], data["Y染色体浓度"])
plt.scatter(data["孕妇BMI"], model.predict(X), color="red")
plt.xlabel("Pregnant BMI")
plt.ylabel("Y chromosome concentration")
plt.title("Pregnant BMI vs Y chromosome concentration (predicted and actual)")
plt.show()
plt.scatter(data["log孕天"], data["Y染色体浓度"])
plt.scatter(data["log孕天"], model.predict(X), color="red")
plt.xlabel("log(Pregnancy days)")
plt.ylabel("Y chromosome concentration")
plt.title("log(Pregnancy days) vs Y chromosome concentration (predicted and actual)")
plt.show()
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """重命名列并进行标准化预处理"""
    column_mapping = {
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
    df = df.rename(columns=column_mapping)
    df["y_clipped"] = df["y_concentration"].clip(lower=1e-6, upper=1 - 1e-6)
    df["y_logit"] = np.log(df["y_clipped"] / (1 - df["y_clipped"]))
    for col in ["bmi", "gestational_weeks"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df
def fit_mixed_model(df: pd.DataFrame):
    """拟合混合效应模型并返回结果"""
    formula = "y_logit ~ bmi + gestational_weeks"
    model = smf.mixedlm(
        formula,
        df,
        groups=df["mother_id"],
        re_formula="~gestational_weeks",
    )
    result = model.fit(reml=False, method="lbfgs", maxiter=2000)
    return result
def evaluate_model(df: pd.DataFrame, result):
    """计算 R² 并生成拟合值与残差"""
    y_obs = df["y_logit"].values
    y_fit = result.fittedvalues.values
    ss_res = np.sum((y_obs - y_fit) ** 2)
    ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    df["fitted"] = result.fittedvalues
    df["residual"] = df["y_logit"] - df["fitted"]
    return r2, df
def plot_diagnostics(df: pd.DataFrame, result):
    """绘制模型诊断图"""
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x="fitted", y="y_logit", data=df)
    min_val, max_val = df["y_logit"].min(), df["y_logit"].max()
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Fitted values")
    plt.ylabel("Observed values")
    plt.title("Fitted vs Observed")
    plt.show()
    re_df = pd.DataFrame(result.random_effects).T
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=re_df)
    plt.title("Random Effects Distribution by Group")
    plt.show()
    plt.figure(figsize=(6, 4))
    sns.histplot(df["residual"], kde=True)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.show()
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x="fitted", y="residual", data=df)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.show()
data = preprocess_data(mod_df)
model_result = fit_mixed_model(data)
print(model_result.summary())
r2, data = evaluate_model(data, model_result)
print(f"R-squared: {r2:.4f}")
plot_diagnostics(data, model_result)
