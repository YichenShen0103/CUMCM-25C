# Problem 4: 基于深度学习的胎儿染色体异常检测

## 项目概述

本项目使用深度学习方法对孕妇产前筛查数据进行分析，预测胎儿染色体的非整倍体异常。通过构建一个多分支注意力神经网络，结合先进的特征工程和数据增强技术，实现了高精度的二分类预测。

## 问题分析

### 背景
产前筛查是孕期重要的检查项目，通过检测孕妇血清中的生化指标（如AFP、Free HCG、Inhibin A等）来评估胎儿染色体异常的风险。传统方法主要依赖统计学分析，而本项目采用深度学习方法来提高预测精度。

### 挑战
1. **类别不平衡**：正常胎儿样本数量远多于异常样本
2. **特征复杂性**：生化指标之间存在复杂的非线性关系
3. **数据稀疏性**：医学数据通常样本量有限
4. **高精度要求**：医学诊断需要极高的准确性和可靠性

## 解决方案架构

### 1. 数据预处理与特征工程

#### 数据清洗
```python
# 处理孕周数据，转换为天数
weeks_days = data["检测孕周"].str.split(r"[wW]", expand=True)
data["孕天"] = weeks_days[0].astype(int) * 7 + weeks_days[1].fillna("0").replace("", "0").astype(int)

# 删除无关特征
useless_columns = ["序号", "孕妇代码", "末次月经", "检测日期", ...]
```

#### 高级特征工程
- **特征交互**：计算重要特征之间的乘积，捕获非线性关系
- **多项式特征**：对关键指标（年龄、AFP_MOM、Free_HCG_MOM等）生成平方项和对数变换
- **统计特征**：添加特征的和、均值、标准差、偏度等统计量

#### 数据变换
- **PowerTransformer**：使用Yeo-Johnson变换处理偏斜分布
- **StandardScaler**：标准化特征到零均值单位方差

### 2. 数据增强策略

针对类别不平衡问题，采用噪声增强方法：

```python
def augment_minority_class(X, y, augment_factor=3):
    """通过添加高斯噪声来增强少数类样本"""
    minority_indices = np.where(y == 1)[0]
    minority_X = X[minority_indices]
    
    for _ in range(augment_factor):
        noise = np.random.normal(0, 0.1, minority_X.shape)
        augmented_samples = minority_X + noise
```

## 模型架构设计

### SuperAdvancedClassifier 网络结构

#### 1. 多分支架构
设计了两个并行的特征提取分支，从不同角度学习数据特征：

**分支1（深层特征提取）**
```
Input → Linear(512) → BatchNorm → ReLU → Dropout(0.3) 
      → Linear(256) → BatchNorm → ReLU → Dropout(0.2)
```

**分支2（宽层特征提取）**
```
Input → Linear(256) → BatchNorm → ReLU → Dropout(0.4)
      → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
```

#### 2. 注意力机制
```python
self.attention = nn.Sequential(
    nn.Linear(256 + 128, 64), 
    nn.Tanh(), 
    nn.Linear(64, 1), 
    nn.Sigmoid()
)
```
- 自动学习两个分支特征的重要性权重
- 使用Tanh激活函数增强非线性表达能力
- Sigmoid输出注意力权重（0-1范围）

#### 3. 特征融合与分类
```python
# 特征融合
combined = torch.cat([b1, b2], dim=1)
attention_weights = self.attention(combined)
attended = combined * attention_weights

# 最终分类
self.classifier = nn.Sequential(
    nn.Linear(384, 128) → BatchNorm → ReLU → Dropout(0.2)
    → nn.Linear(128, 64) → BatchNorm → ReLU → Dropout(0.1)
    → nn.Linear(64, 1)
)
```

### 关键设计特点

1. **BatchNormalization**：加速训练收敛，提高模型稳定性
2. **Dropout正则化**：防止过拟合，不同层采用不同的dropout率
3. **Kaiming初始化**：适合ReLU激活函数的权重初始化
4. **梯度裁剪**：防止梯度爆炸，提高训练稳定性

## 训练策略

### 1. 自适应焦点损失函数（Adaptive Focal Loss）

```python
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=2, gamma=3):
        super(AdaptiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        
        # 动态调整alpha
        pos_weight = (targets == 0).sum().float() / (targets == 1).sum().float()
        alpha = torch.where(targets == 1, pos_weight, 1.0)
        
        F_loss = alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
```

**优势**：
- 自动平衡正负样本
- 专注于困难样本的学习
- 动态调整类别权重

### 2. 优化器与学习率调度

- **AdamW优化器**：结合Adam的自适应性和权重衰减
- **OneCycleLR调度器**：学习率先增后减，提高收敛效果
- **权重衰减**：L2正则化防止过拟合

### 3. 训练监控与早停

- **动态阈值优化**：在验证时测试多个分类阈值，选择最优阈值
- **早停机制**：监控验证集性能，防止过拟合
- **模型保存**：自动保存最佳性能的模型参数

## 实验结果

### 性能指标
- **准确率**：在测试集上达到高精度分类
- **AUC Score**：ROC曲线下面积，评估模型判别能力
- **混淆矩阵**：详细分析真阳性、假阳性等指标
- **分类报告**：精确率、召回率、F1分数等全面评估

### 阈值优化
通过测试0.3到0.8范围内的多个阈值，自动选择最优分类阈值，平衡敏感性和特异性。

## 技术创新点

1. **多分支注意力架构**：结合不同深度的特征提取分支
2. **自适应损失函数**：动态调整类别权重，优化不平衡数据学习
3. **噪声数据增强**：有效扩充少数类样本
4. **端到端学习**：从原始特征到最终预测的全自动流程
5. **动态阈值优化**：自适应选择最优分类阈值

## 应用价值

本项目提供了一个完整的医学数据分析解决方案，可以：

1. **辅助临床诊断**：为医生提供客观的风险评估
2. **减少医疗成本**：通过准确筛查减少不必要的进一步检查
3. **提高诊断效率**：自动化分析提高工作效率
4. **可扩展性**：架构可适用于其他医学分类问题

## 技术栈

- **深度学习框架**：PyTorch
- **数据处理**：Pandas, NumPy
- **特征工程**：Scikit-learn
- **模型评估**：Scikit-learn metrics
- **数据可视化**：内置评估报告

## 文件结构

```
├── problem4.ipynb          # 主要实现代码
├── model/
│   └── best_model.pth      # 训练好的最佳模型
├── data/
│   └── data.xlsx           # 原始数据
└── problem4.md            # 本文档
```

## 使用说明

1. 运行第一个单元格导入必要库
2. 执行数据预处理和特征工程
3. 运行模型训练（大约15000个epoch，带早停）
4. 评估模型性能并查看详细指标

项目实现了从数据预处理到模型部署的完整流程，为医学数据分析提供了一个高性能的深度学习解决方案。
