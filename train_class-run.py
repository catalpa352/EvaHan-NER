from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
from useful_class import train_class
#测试调用：
tokenizer = AutoTokenizer.from_pretrained("hsc748NLP/GujiRoBERTa_jian_fan")
model = AutoModelForMaskedLM.from_pretrained("hsc748NLP/GujiRoBERTa_jian_fan")
#原始文本地址：
file_path = "EvaHan2025_traingdata/trainset_B.txt"

train_entity=train_class(model,tokenizer,file_path)

train_entity.train(35,64,5,5e-5)

# #模型保存地址：
output_dir = "saved_model_trainsetB"
train_entity.save_model(output_dir)
