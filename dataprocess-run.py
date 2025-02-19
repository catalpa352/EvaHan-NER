from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
from useful_class import dataprocess

# tokenizer = AutoTokenizer.from_pretrained("hsc748NLP/GujiRoBERTa_jian_fan")
# model = AutoModelForMaskedLM.from_pretrained("hsc748NLP/GujiRoBERTa_jian_fan")
file_path="EvaHan2025_traingdata/trainset_A_自己修改的.txt"
dataprocessor = dataprocess(35)
# texts_id,attention_mask,label_ids,texts,tags=dataprocessor.process_data(file_path, tokenizer)
# print(texts_id[0].shape)
# print(attention_mask[0].shape)
# print(label_ids[0].shape)
# print(len(texts[0]),texts[0])
# print(len(tags[0]),tags[0])
#
# print("——————————————————————————————————")
# print(len(texts[5713]))
# print(len(texts[5714]))
lines=dataprocessor.read_conll_file(file_path)
print(type(lines),len(lines))
texts, tags = dataprocessor.split_into_segments2(lines)
print(type(texts),len(texts))
max=len(texts[0])
max_index=0
for i in range(len(texts)):
    if len(texts[i])>=max:
        max=len(texts[i])
        max_index=i
print("最大长度为：",max)
print(texts[max_index])
print(tags[max_index])



min=len(texts[0])
min_index=0
for i in range(len(texts)):
    if len(texts[i])<=max:
        min=len(texts[i])
        min_index=i
print("最小长度为：",min)
print(texts[min_index])
print(tags[min_index])

print(texts[min_index-1])
print(tags[min_index-1])

