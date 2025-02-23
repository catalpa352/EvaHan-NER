from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from test_class import test
# 定义保存模型的目录路径
output_dir = "./saved_model_trainsetB"
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(output_dir)
# 加载模型
model = AutoModelForMaskedLM.from_pretrained(output_dir)
file_path = "EvaHan2025_testdata/testset_B.txt"
test1 = test(output_dir)





result_file_path="result2.txt"
test1.predict(35,file_path,64,result_file_path)

# output_file_path= "real_sign_result3.txt"
# test1.map_predictions_to_labels(result_file_path,output_file_path)
