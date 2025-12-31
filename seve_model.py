import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle  # 用来保存模型和output_uniques变量

# 读取数据集，并将字符编码指定为gbk，防止中文报错
penguin_df = pd.read_csv('penguins-chinese.csv', encoding='gbk')
# 删除缺失值所在的行
penguin_df.dropna(inplace=True)
# 将企鹅的种类定义为目标输出变量
output = penguin_df['企鹅的种类']
# 选择特征列（企鹅栖息的岛屿、喙的长度、喙的深度、翅膀的长度、身体质量、性别）
features = penguin_df[['企鹅栖息的岛屿', '喙的长度', '喙的深度', '翅膀的长度', '身体质量', '性别']]
# 对特征列进行独热编码
features = pd.get_dummies(features)
# 将目标输出变量转换为离散数值
output_codes, output_uniques = pd.factorize(output)

# 划分训练集（80%）和测试集（20%）
x_train, x_test, y_train, y_test = train_test_split(features, output_codes, train_size=0.8)

# 构建并训练随机森林分类器
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
# 测试集预测
y_pred = rfc.predict(x_test)
# 计算预测准确率
score = accuracy_score(y_test, y_pred)

# 保存模型到文件
with open('rfc_model.pkl', 'wb') as f:
    pickle.dump(rfc, f)
# 保存类别映射关系到文件
with open('output_uniques.pkl', 'wb') as f:
    pickle.dump(output_uniques, f)

print('保存成功，已生成相关文件。')
