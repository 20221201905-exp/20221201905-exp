实验一 线性回归实验报告

序号：24 姓名：王凯文 学号：20221201905

一、实验目的

深入理解线性回归模型的基本原理、假设及其在实际问题中的应用。
熟练运用 scikit-learn 机器学习库，掌握数据加载、预处理、模型训练、评估和结果可视化的完整流程。
通过对电力输出数据集的分析，培养数据洞察能力，理解特征变量与目标变量间的复杂关系。
二、实验环境

操作系统: Windows 10
开发工具: PyCharm
编程语言: Python 3.9
所需库:
numpy: 高性能数值计算库。
pandas: 强大的数据结构和数据分析工具。
matplotlib: 基础绘图库，用于创建静态、动态、交互式可视化。
seaborn: 基于matplotlib的数据可视化库，提供更美观的统计图。
scikit-learn (>=0.18): 机器学习核心库，包含各种分类、回归、聚类算法。
三、实验内容

本实验的核心是基于 Folds5x2_pp.csv 数据集构建一个线性回归模型，以预测发电厂的净小时电能输出 (PE)。该数据集共包含9568个样本，每个样本拥有以下5个特征：

AT: 环境温度 (Ambient Temperature)
V: 废气真空度 (Exhaust Vacuum)
AP: 大气压力 (Atmospheric Pressure)
RH: 相对湿度 (Relative Humidity)
PE: 净小时电能输出 (Net Hourly Electrical Energy Output, 目标变量)

​
  为对应特征的回归系数。实验将数据集按3:1的比例划分为训练集和测试集，利用 scikit-learn 对模型进行训练，并全面评估其性能。

四、实验步骤

环境搭建

在 Windows 10 系统中，确保已安装 Python 3.9。通过命令行工具（如PowerShell或CMD）执行以下命令来安装所有必要的Python库：

Bash

pip install numpy pandas matplotlib seaborn scikit-learn
数据预处理

此阶段主要完成数据集的加载、特征与目标变量的分离，以及数据集的训练集与测试集划分。

Python

# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以确保结果可重复性
np.random.seed(1)

def getTrainSetAndTestSet(DataPath):
    """读取数据集并划分为训练集和测试集"""
    data = pd.read_csv(DataPath)
    X = data[['AT', 'V', 'AP', 'RH']]  # 提取特征变量
    y = data['PE']  # 提取目标变量
    # 按照3:1的比例划分训练集和测试集，test_size=0.25表示25%的数据用于测试
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    return X_train, X_test, y_train, y_test

# 加载数据
data_path = 'Folds5x2_pp.csv' # 请确保此CSV文件位于代码运行目录或提供完整路径
X_train, X_test, y_train, y_test = getTrainSetAndTestSet(data_path)

# 显示数据集基本信息
print("数据集总样本数:", len(pd.read_csv(data_path)))
print("训练集形状 (样本数, 特征数):", X_train.shape)
print("测试集形状 (样本数, 特征数):", X_test.shape)
print("\n训练集描述统计:")
print(pd.concat([X_train, y_train], axis=1).describe())
print("\n训练集前5行:")
print(pd.concat([X_train, y_train], axis=1).head())
数据预处理分析：

数据集规模: 原始数据集共9568个样本，经划分后，训练集包含7176个样本，测试集包含2392个样本，这样的规模对于线性回归模型训练是足够的。
特征与目标: 明确选取了AT、V、AP、RH作为自变量，PE作为因变量，符合实验目的。
划分策略: 采用标准的3:1随机划分比例，并设置 random_state=1 确保结果可复现，这有助于模型在训练过程中学习数据模式，并在独立测试集上评估其泛化能力。
描述统计: describe() 函数的输出提供了各特征及目标变量的均值、标准差、四分位数、最小值和最大值。这有助于我们初步了解数据的中心趋势、离散程度以及是否存在异常值。例如，PE的范围在420.26到495.76之间，具有一定的波动性。
探索性数据分析 (EDA)

通过可视化手段深入探索数据的内在结构，包括特征的分布、特征之间的相互关系以及特征与目标变量 PE 之间的关联。

Python

# 特征分布直方图
plt.figure(figsize=(12, 8))
for i, column in enumerate(['AT', 'V', 'AP', 'RH'], 1):
    plt.subplot(2, 2, i) # 2行2列子图布局
    sns.histplot(X_train[column], kde=True, color='skyblue') # 绘制带核密度估计的直方图
    plt.title(f'特征 {column} 的分布')
    plt.xlabel(column)
    plt.ylabel('频率')
plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
plt.show()

# 特征与PE的散点图
plt.figure(figsize=(12, 8))
for i, column in enumerate(['AT', 'V', 'AP', 'RH'], 1):
    plt.subplot(2, 2, i)
    plt.scatter(X_train[column], y_train, alpha=0.3, color='green') # 绘制散点图
    plt.xlabel(column)
    plt.ylabel('PE (输出功率)')
    plt.title(f'{column} 与 PE 的关系')
plt.tight_layout()
plt.show()

# 相关性热图
plt.figure(figsize=(8, 6))
correlation_matrix = pd.concat([X_train, y_train], axis=1).corr() # 计算所有特征和目标变量间的相关性矩阵
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f') # 绘制热力图，显示相关系数
plt.title('特征与PE的相关性矩阵')
plt.show()
探索性数据分析结果：

特征分布: 对各特征的直方图分析显示，AT（温度）和 V（催化剂）可能呈现出近似双峰的分布，而 AP（大气压力）则接近于正态分布，RH（相对湿度）的分布范围较广。这些分布特性提示我们，数据可能存在多种操作条件或环境因素，这对于后续模型解释具有重要意义。
特征与PE关系散点图: 散点图直观地展示了各特征与目标变量 PE 之间的初步关系。可以观察到 AT 和 V 与 PE 之间存在较为明显的负相关趋势，即随着温度或催化剂的增加，输出功率呈下降趋势。而 AP 和 RH 与 PE 的线性关系相对较弱，散点分布更为分散。
相关性分析热图: 相关性热图定量地揭示了特征之间以及特征与目标变量之间的线性相关强度。我们注意到 AT 与 PE 具有较高的负相关系数（如 -0.95），强烈表明温度是影响输出功率的关键因素。同时，特征之间也可能存在一定的多重共线性（例如 AT 与 V 之间可能存在中等程度的相关性），这在多变量线性回归中需要引起关注，可能影响系数的解释或模型的稳定性，但对模型整体预测能力影响较小。
训练线性回归模型

使用 scikit-learn 中的 LinearRegression 类来构建并训练模型。

Python

def TrainLinearRegression(X_train, y_train):
    """训练线性回归模型并返回模型对象"""
    linreg = LinearRegression() # 实例化线性回归模型
    linreg.fit(X_train, y_train) # 在训练数据上拟合模型
    print("模型系数 (θ1:AT, θ2:V, θ3:AP, θ4:RH):", linreg.coef_)
    print("截距 (θ0):", linreg.intercept_)
    return linreg

# 训练模型
linreg = TrainLinearRegression(X_train, y_train)
输出结果：

模型系数 (θ1:AT, θ2:V, θ3:AP, θ4:RH): [-1.97376045 -0.23229086  0.0693515  -0.15806957]
截距 (θ0): 447.0629709868725
模型训练分析：

模型参数解读: 训练得到的模型系数和截距揭示了各特征对 PE 的定量影响。
AT 的系数约为 -1.97：表明在其他条件不变的情况下，环境温度每升高1个单位，输出功率 PE 大约下降1.97个单位。这与EDA中的负相关观察一致。
V 的系数约为 -0.23：废气真空度每增加1个单位，PE 大约下降0.23个单位。
AP 的系数约为 0.07：大气压力每增加1个单位，PE 略微增加约0.07个单位，影响较小。
RH 的系数约为 -0.16：相对湿度每增加1个单位，PE 大约下降0.16个单位。
截距 θ 
0
​
  (447.06)：表示当所有特征（AT, V, AP, RH）均为零时，PE 的预测值。但在实际物理意义上，特征值很少为零，因此截距更多是模型拟合的数学结果。
训练过程: scikit-learn 的 LinearRegression 默认采用普通最小二乘法 (OLS) 来优化损失函数，寻找使残差平方和最小化的参数。这种方法计算效率高，在数据规模适中时表现良好。
模型评估

使用多种常用的回归评估指标（均方误差、均绝对误差、R²分数）来量化模型在测试集上的预测性能。

Python

def EvaluationModel(linreg, X_test, y_test):
    """评估模型性能"""
    y_pred = linreg.predict(X_test) # 在测试集上进行预测
    mse = mean_squared_error(y_test, y_pred) # 计算均方误差 (MSE)
    rmse = np.sqrt(mse) # 计算均方根误差 (RMSE)
    mae = mean_absolute_error(y_test, y_pred) # 计算均绝对误差 (MAE)
    r2 = r2_score(y_test, y_pred) # 计算R²分数
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"均绝对误差 (MAE): {mae:.2f}")
    print(f"R² 分数: {r2:.4f}")
    return y_pred

# 评估模型
y_pred = EvaluationModel(linreg, X_test, y_test)
输出结果：

均方误差 (MSE): 20.08
均方根误差 (RMSE): 4.48
均绝对误差 (MAE): 3.61
R² 分数: 0.9317
模型评估结果分析：

均方误差 (MSE): 20.08。MSE 衡量了预测值与实际值之间平方差的平均值。值越小，表示模型预测精度越高。
均方根误差 (RMSE): 4.48。RMSE 是MSE的平方根，与目标变量 PE 的单位相同，因此更直观地反映了模型预测的平均误差大小。4.48的RMSE表示平均而言，模型的预测值与实际值相差约4.48个单位的电力输出。
均绝对误差 (MAE): 3.61。MAE 衡量了预测值与实际值之间绝对差的平均值。相较于MSE和RMSE，MAE对异常值不那么敏感，能更直接地反映平均预测误差。3.61的MAE表明模型平均预测误差为3.61。
R² 分数: 0.9317。R 
2
  分数（决定系数）表示模型解释了目标变量 PE 变异的比例。0.9317的R²分数意味着模型能够解释大约93.17%的输出功率变化，这是一个非常高的值，表明线性回归模型对该数据集的拟合效果极佳，具有很强的解释能力和预测能力。
综合分析: MSE、RMSE和MAE均处于较低水平，且R²分数非常接近1（高达0.9317），这强烈表明训练出的线性回归模型具有出色的泛化能力，能够准确地预测测试集上的电力输出 PE。

结果可视化

通过绘制散点图、残差图和残差分布图，直观地分析模型的预测效果和残差特性。

Python

def Visualization(y_test, y_pred):
    """可视化预测结果"""
    # 预测值 vs 实际值散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='预测点')
    # 绘制对角线 y=x，表示理想情况下的完美预测
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='理想预测线 ($y=x$)')
    plt.xlabel('实际输出功率 (Actual PE)')
    plt.ylabel('预测输出功率 (Predicted PE)')
    plt.title('预测值 vs 实际值散点图')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 残差图：残差与预测值的关系
    residuals = y_test - y_pred # 计算残差
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color='purple', label='残差')
    plt.axhline(y=0, color='red', linestyle='--', lw=2, label='零残差线') # 绘制y=0的参考线
    plt.xlabel('预测输出功率 (Predicted PE)')
    plt.ylabel('残差 (Residuals)')
    plt.title('残差图：预测值与残差的关系')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 残差分布直方图
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='orange', bins=30) # 绘制带核密度估计的残差直方图
    plt.xlabel('残差 (Residuals)')
    plt.title('残差分布图')
    plt.tight_layout()
    plt.show()

# 可视化
Visualization(y_test, y_pred)
可视化分析：

预测值 vs 实际值散点图: 图中蓝色的散点大部分紧密地围绕着黑色的 y=x 对角线分布，这直观地表明模型的预测值与实际值高度一致，预测精度非常高。只有极少数点略微偏离，进一步印证了模型评估指标的优异结果。
残差图: 残差图显示了预测值与实际值之间的误差（残差）与预测值之间的关系。理想情况下，残差应随机分布在零轴（红色虚线）上下，没有明显的模式（如U形、V形或扇形）。本实验的残差图表现出良好的随机性，大多数残差点均匀地分布在零线附近，没有明显的趋势或结构，这支持了线性回归模型假设（如误差项独立同分布）的合理性。
残差分布图: 残差的直方图显示，其分布形态大致呈钟形，并且中心接近于零，这表明残差近似服从正态分布。残差的正态性是线性回归模型的重要假设之一。虽然并非严格的正态分布（可能有轻微的偏态或峰度），但整体趋势良好，进一步验证了模型的拟合效果。
五、实验结果与分析

本次线性回归实验成功构建并评估了基于温度、催化剂、湿度和压强预测电厂输出功率的线性模型。

实验结果总结：

模型拟合优异: 通过对训练数据的拟合，模型学得了各特征与输出功率之间的线性关系。回归系数揭示了各特征对PE的贡献方向和强度：温度和废气真空度与输出功率呈负相关，且影响显著；大气压力和相对湿度的影响相对较小，前者为正相关，后者为负相关。
评估指标卓越: 在独立的测试集上，模型的性能表现突出。
均方误差 (MSE) 20.08、均方根误差 (RMSE) 4.48、均绝对误差 (MAE) 3.61，均处于较低水平，表明模型的平均预测误差很小。
R² 分数高达0.9317，意味着该模型能够解释约93.17%的输出功率变异，证明了模型对数据的高度拟合能力和强大的解释力。
可视化验证:
“预测值 vs 实际值”散点图直观地展示了预测值与实际值的极高一致性，绝大多数点紧密聚集在理想的 y=x 对角线周围。
残差图显示残差随机分布在零轴附近，没有明显的模式，这表明模型捕捉了数据中的主要线性关系，且误差项独立且方差恒定。
残差分布图近似正态分布，进一步支持了模型假设的合理性。
局限性与展望：
尽管本次实验取得了令人满意的结果，但也存在一些值得探讨的局限性：

线性假设: 线性回归模型的核心假设是特征与目标变量之间存在线性关系。如果数据中存在显著的非线性关系，线性模型可能无法完全捕捉这些复杂模式，从而限制模型的进一步优化。
多重共线性: EDA中观察到特征间可能存在一定的相关性（如AT和V），这可能导致模型系数的解释性受到影响，尽管对整体预测精度影响不大。
特征工程: 尽管模型性能良好，但并未进行深入的特征工程，例如特征标准化（尽管线性回归对特征缩放不敏感，但对梯度下降优化器有益）、异常值处理或引入多项式特征等，这些步骤可能在某些情况下进一步提升模型性能。
结论:
本次实验成功地运用线性回归模型对发电厂输出功率进行了有效预测。通过规范的数据预处理、全面的探索性数据分析、严谨的模型训练和多维度评估，我们不仅掌握了线性回归的基本原理和 scikit-learn 的使用方法，更深刻理解了如何从数据中提取信息，建立预测模型，并对其性能进行客观评价。实验结果充分证明了线性回归在处理此类多变量回归问题中的强大有效性。

展望未来，为了进一步提升模型的泛化能力和预测精度，可以考虑：

尝试非线性模型: 如决策树回归、支持向量机回归或集成学习模型（如随机森林、梯度提升树），以捕捉数据中潜在的非线性关系。
正则化技术: 引入L1（Lasso）或L2（Ridge）正则化，以应对多重共线性问题，并提高模型的泛化能力，防止过拟合。
更细致的特征工程: 对数据进行深入分析，可能包括交互项的构建、非线性变换或更复杂的特征组合。
本次实验为后续更复杂的机器学习算法学习和实际项目应用奠定了坚实的基础。