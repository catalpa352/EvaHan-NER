from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from test_class import test
# 定义保存模型的目录路径
output_dir = "./saved_model"
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(output_dir)
# 加载模型
model = AutoModelForMaskedLM.from_pretrained(output_dir)
file_path = "EvaHan2025_traingdata/trainset_A_自己修改的.txt"
test1 = test(output_dir)

report=test1.test(35, file_path, 64)
# 将字典转换为DataFrame
df_report = pd.DataFrame(report).transpose()

# 指定保存报告的文件路径
output_file_path = 'classification_report_固定长度和标点符号切分.csv'

# 将DataFrame保存为CSV文件
df_report.to_csv(output_file_path, index=True)

print(f"Classification report has been saved to {output_file_path}")
