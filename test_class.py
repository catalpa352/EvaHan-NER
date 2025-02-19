from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from useful_class import dataprocess
from torch.utils.data import DataLoader
from tqdm import tqdm
#测试类：
class test:
    def __init__(self, model_file_path):
        self.model = AutoModelForMaskedLM.from_pretrained(model_file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_file_path)

    #预测结果，然后计算acc,recall,F1,打印出报告：
    def test(self, max_length, test_file_path, batch_size):
        # 自定义 collate_fn
        def collate_fn(batch):
            input_ids = [item["input_ids"] for item in batch]
            attention_mask = [item["attention_mask"] for item in batch]
            labels = [item["labels"] for item in batch]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        model = self.model
        # 确保模型处于评估模式
        model.eval()
        # 准备测试数据集
        test_dataset = dataprocess(max_length)
        test_dataset.process_data(test_file_path, self.tokenizer)  # 假设你有另一个文件路径用于测试

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        all_predictions = []
        all_labels = []

        with torch.no_grad():  # 不需要计算梯度
            # for batch in test_loader:
            for batch in tqdm(test_loader, desc="Processing batches", unit="batch"):
                input_ids = torch.stack(batch["input_ids"]).squeeze(1).to(device)
                attention_mask = torch.stack(batch["attention_mask"]).squeeze(1).to(device)
                labels = torch.stack(batch["labels"]).squeeze(1).to(device)
                # print("labels值与形状：",labels, labels.shape)
                # 获取模型预测结果
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                # 将logits转换为预测标签[2,15,30546]——》【2，15】
                predictions = torch.argmax(logits, dim=-1)
                # print("predictions的值与形状",predictions, predictions.shape)
                # 移除padding标记（通常是0），以便于后续评估
                for pred, label in zip(predictions, labels):
                    # print(pred.shape, label.shape)
                    # print(pred, label)
                    valid_preds = pred[label != 0]
                    valid_labels = label[label != 0]
                    # print(valid_preds.shape,valid_labels.shape)
                    # print(valid_preds,valid_labels)
                    all_predictions.extend(valid_preds.cpu().numpy())
                    all_labels.extend(valid_labels.cpu().numpy())

        # 计算评估指标
        from sklearn.metrics import classification_report
        report = classification_report(all_labels, all_predictions,output_dict=True)

        print(report)
        return report


    #预测结果，输出预测结果到指定路径：
    def predict(self, max_length, test_file_path, batch_size, output_file_path):
        # 自定义 collate_fn
        def collate_fn(batch):
            input_ids = [item["input_ids"] for item in batch]
            attention_mask = [item["attention_mask"] for item in batch]
            labels = [item["labels"] for item in batch]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        model = self.model
        # 确保模型处于评估模式
        model.eval()
        # 准备测试数据集
        test_dataset = dataprocess(max_length)
        test_dataset.process_data(test_file_path, self.tokenizer)  # 假设你有另一个文件路径用于测试

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        all_predictions = []

        with torch.no_grad():  # 不需要计算梯度
            for batch in tqdm(test_loader, desc="Processing batches", unit="batch"):
                input_ids = torch.stack(batch["input_ids"]).squeeze(1).to(device)
                attention_mask = torch.stack(batch["attention_mask"]).squeeze(1).to(device)
                labels = torch.stack(batch["labels"]).squeeze(1).to(device)

                # 获取模型预测结果
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                # 将logits转换为预测标签[2,15,30546]——》【2，15】
                predictions = torch.argmax(logits, dim=-1)

                # 移除padding标记（通常是0），以便于后续评估
                for pred, label in zip(predictions, labels):
                    valid_preds = pred[label != 0]  # 只保留非padding部分的预测结果
                    all_predictions.extend(valid_preds.cpu().numpy())

        # 将预测结果逐行保存到txt文件
        with open(output_file_path, "w") as f:
            for pred in all_predictions:
                f.write(f"{pred}\n")  # 每个预测结果占一行

        print(f"Predictions saved to {output_file_path}")
        return all_predictions


    #将预测的数字标签转换成有意义的标签
    def map_predictions_to_labels(self,input_file_path, output_file_path):
        """
        将包含预测结果的txt文件中的数字映射回标签，并保存到另一个文件中。

        参数:
        input_file_path (str): 输入文件路径，包含预测结果的数字。
        output_file_path (str): 输出文件路径，保存映射后的标签。
        """
        # 定义标签列表
        labels = [
            "O", "B-NR", "E-NR", "M-NR", "S-T", "S-NG", "B-NS", "E-NS", "S-NR",
            "B-T", "M-T", "E-T", "S-NS", "S-NO", "B-NO", "M-NO", "E-NO", "B-NG",
            "E-NG", "B-NB", "E-NB", "M-NS", "S-NB", "M-NB", "M-NG"
        ]

        # 反转映射，从数字映射回标签
        id_to_label = {idx + 1: label for idx, label in enumerate(labels)}

        # 读取输入文件并映射标签
        with open(input_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
            for line in input_file:
                # 去掉换行符并转换为整数
                pred_id = int(line.strip())
                # 映射回标签
                if pred_id in id_to_label:
                    mapped_label = id_to_label[pred_id]
                else:
                    mapped_label = "Unknown"  # 处理未知的ID
                # 将映射后的标签写入输出文件
                output_file.write(f"{mapped_label}\n")

        print(f"映射后的标签已保存到 {output_file_path}")



