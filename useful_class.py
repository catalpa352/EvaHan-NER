from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

#数据预处理类
class dataprocess(Dataset):
    def __init__(self, max_length=0):
        self.max_length = max_length
        self.texts_id = None
        self.texts = None
        self.attention_mask = None
        self.label_ids = None
        self.labels = None

    # 读取文本
    def read_conll_file(self, file_path):
        """
        读取CoNLL格式的数据文件，并返回所有行组成的列表。

        参数:
            file_path (str): 文件路径。

        返回:
            lines (list): 文件中每一行的内容列表。
        """
        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        return [line.strip() for line in lines if line.strip()]  # 去除空行

    # 根据固定长度切分数据与标签
    def split_into_segments(self, lines, max_length=28):
        """
        将CoNLL格式的行列表转换为训练数据格式，按指定字符数进行切分。

        参数:
            lines (list): CoNLL格式的行列表。
            max_length (int): 每个片段的最大字符长度，默认28。

        返回:
            train_texts (list): 句子列表。
            train_labels (list): 标签列表。
        """
        train_texts = []
        train_labels = []

        current_segment = []
        current_labels = []
        current_length = 0

        for line in lines:
            word, label = line.split("\t")

            if current_length + len(word) >= max_length + 1:
                if current_segment:
                    train_texts.append("".join(current_segment))
                    train_labels.append(current_labels)
                    current_segment = []
                    current_labels = []
                    current_length = 0

            if len(word) >= max_length + 1:
                train_texts.append(word)
                train_labels.append([label])
            else:
                current_segment.append(word)
                current_labels.append(label)
                current_length += len(word)

        if current_segment:
            train_texts.append("".join(current_segment))
            train_labels.append(current_labels)

        return train_texts, train_labels

    # 根据固定长度和标点符号切分数据与标签：
    def split_into_segments2(self, lines, max_length=28):
        """
        将CoNLL格式的行列表转换为训练数据格式，根据标点切分句子，同时确保实体完整性。

        参数:
            lines (list): CoNLL格式的行列表。
            max_length (int): 每个片段的最大字符长度，默认28。

        返回:
            train_texts (list): 句子列表。
            train_labels (list): 标签列表。
        """
        train_texts = []
        train_labels = []

        current_segment = []
        current_labels = []
        current_length = 0

        for line in lines:
            if not line.strip():  # 跳过空行
                continue

            word, label = line.split("\t")

            # 如果当前句子长度超过限制，或者遇到标点符号（句号、问号、感叹号），则切分句子
            if (current_length + len(word) >= max_length + 1) or (word in {"。", "？", "！"}):
                if current_segment:
                    train_texts.append("".join(current_segment))
                    train_labels.append(current_labels)
                    current_segment = []
                    current_labels = []
                    current_length = 0

            # 如果单个词长度超过限制，单独作为一个句子
            if len(word) >= max_length + 1:
                train_texts.append(word)
                train_labels.append([label])
            else:
                current_segment.append(word)
                current_labels.append(label)
                current_length += len(word)

        # 处理最后一个句子
        if current_segment:
            train_texts.append("".join(current_segment))
            train_labels.append(current_labels)

        return train_texts, train_labels

    #  根据固定长度和实体标签切分数据与标签：
    def split_based_on_entity_boundaries(self, lines, max_length=20):
        """
        将CoNLL格式的行列表转换为训练数据格式，基于实体边界进行切分，确保实体完整性。

        参数:
            lines (list): CoNLL格式的行列表。
            max_length (int): 每个片段的最大字符长度，默认28。

        返回:
            train_texts (list): 句子列表。
            train_labels (list): 标签列表。
        """
        train_texts = []
        train_labels = []

        current_segment = []
        current_labels = []
        current_length = 0

        in_entity = False  # 标记是否处于实体内部
        entity_start = False  # 标记是否是实体开始

        for line in lines:
            if not line.strip():  # 跳过空行
                continue

            word, label = line.split("\t")

            # 如果当前词属于某个实体（即非'O'），设置in_entity标志
            if label != 'O':
                if not in_entity:
                    entity_start = True
                in_entity = True
            else:
                if in_entity:
                    # 当从实体转换到非实体时，如果超出长度限制，则在此处切割
                    if current_length >= max_length:
                        train_texts.append("".join(current_segment))
                        train_labels.append(current_labels)
                        current_segment = [word]
                        current_labels = [label]
                        current_length = len(word)
                        in_entity = False
                        continue
                in_entity = False

            # 如果添加当前词后超过最大长度，并且不在实体中或位于实体开头，则进行切分
            if current_length + len(word) > max_length and (not in_entity or entity_start):
                train_texts.append("".join(current_segment))
                train_labels.append(current_labels)
                current_segment = [word]
                current_labels = [label]
                current_length = len(word)
            else:
                current_segment.append(word)
                current_labels.append(label)
                current_length += len(word)

            if entity_start:
                entity_start = False

        if current_segment:
            train_texts.append("".join(current_segment))
            train_labels.append(current_labels)

        return train_texts, train_labels

    #动态窗口分割：
    def dynamic_window_split(self,lines, max_length=28, min_length=10, stop_words=None):
        """
        动态窗口分割实现
        参数:
            lines: CoNLL格式行列表（每行格式：词\t标签）
            max_length: 最大字符长度（硬限制）
            min_length: 最小语义单元长度（建议值）
            stop_words: 建议切分点词列表（可配置）
        返回:
            (文本列表, 标签列表)
        """
        # 检查输入是否为空
        if lines is None:
            raise ValueError("输入数据不能为 None，请检查数据加载流程")
        if not isinstance(lines, (list, tuple)):
            raise TypeError("输入数据必须是列表或元组")

        # 默认停用词列表（可根据具体任务调整）
        default_stops = {"的", "了", "和", "在", "是", "就", "而", "与", "及", "或", "等", "后", "并"}
        stop_words = stop_words or default_stops

        train_texts = []
        train_labels = []

        current_words = []
        current_labels = []
        current_len = 0

        # 预扫描获取词列表
        word_list = []
        for line in lines:
            if line.strip():
                word, label = line.split("\t")
                word_list.append((word, label))

        pointer = 0
        while pointer < len(word_list):
            word, label = word_list[pointer]
            word_len = len(word)

            # 情况1：遇到超长词（强制单独成段）
            if word_len > max_length:
                if current_words:  # 先处理已有缓存
                    train_texts.append("".join(current_words))
                    train_labels.append(current_labels)
                    current_words = []
                    current_labels = []
                    current_len = 0
                train_texts.append(word)
                train_labels.append([label])
                pointer += 1
                continue

            # 情况2：可以正常添加
            if current_len + word_len <= max_length:
                current_words.append(word)
                current_labels.append(label)
                current_len += word_len
                pointer += 1
            else:
                # 动态寻找最佳切分点（从后往前扫描）
                split_pos = None
                look_back = min(len(current_words), 10)  # 最多回看10个词

                # 策略1：优先在停用词位置切分
                for i in range(len(current_words) - 1, max(-1, len(current_words) - look_back - 1), -1):
                    if current_words[i] in stop_words and (
                            current_len - sum(len(w) for w in current_words[i + 1:])) >= min_length:
                        split_pos = i + 1
                        break

                # 策略2：次优选择（逗号等符号）
                if split_pos is None:
                    for i in range(len(current_words) - 1, max(-1, len(current_words) - look_back - 1), -1):
                        if current_words[i] in {"，", "；","。", "？", "！"}:
                            split_pos = i + 1
                            break

                # 策略3：保底选择（满足最小长度）
                if split_pos is None:
                    for i in range(len(current_words) - 1, 0, -1):
                        if (current_len - sum(len(w) for w in current_words[i:])) >= min_length:
                            split_pos = i
                            break

                # 执行切分
                if split_pos is not None and split_pos > 0:
                    # 处理前半部分
                    seg_words = current_words[:split_pos]
                    seg_labels = current_labels[:split_pos]
                    train_texts.append("".join(seg_words))
                    train_labels.append(seg_labels)

                    # 保留后半部分
                    current_words = current_words[split_pos:]
                    current_labels = current_labels[split_pos:]
                    current_len = sum(len(w) for w in current_words)
                else:
                    # 无法找到合适切分点，强制切分
                    train_texts.append("".join(current_words))
                    train_labels.append(current_labels)
                    current_words = []
                    current_labels = []
                    current_len = 0

        # 处理剩余部分
        if current_words:
            train_texts.append("".join(current_words))
            train_labels.append(current_labels)

        return train_texts, train_labels

    #混合策略分割：
    def hybrid_split(self,lines, max_length=28, min_length=10, stop_words=None):
        """
        混合策略分割实现
        参数:
            lines: CoNLL格式行列表（每行格式：词\t标签）
            max_length: 最大字符长度（硬限制）
            min_length: 最小语义单元长度（建议值）
            stop_words: 建议切分点词列表（可配置）
        返回:
            (文本列表, 标签列表)
        """
        # 默认停用词列表（可根据具体任务调整）
        default_stops = {"的", "了", "和", "在", "是", "就", "而", "与", "及", "或", "等", "后", "并"}
        stop_words = stop_words or default_stops

        # 检查输入是否为空
        if lines is None:
            raise ValueError("输入数据不能为 None，请检查数据加载流程")
        if not isinstance(lines, (list, tuple)):
            raise TypeError("输入数据必须是列表或元组")

        train_texts = []
        train_labels = []

        current_words = []
        current_labels = []
        current_len = 0

        # 预扫描获取词列表
        word_list = []
        for line in lines:
            if line.strip():
                word, label = line.split("\t")
                word_list.append((word, label))

        pointer = 0
        while pointer < len(word_list):
            word, label = word_list[pointer]
            word_len = len(word)

            # 情况1：遇到超长词（强制单独成段）
            if word_len > max_length:
                if current_words:  # 先处理已有缓存
                    train_texts.append("".join(current_words))
                    train_labels.append(current_labels)
                    current_words = []
                    current_labels = []
                    current_len = 0
                train_texts.append(word)
                train_labels.append([label])
                pointer += 1
                continue

            # 情况2：可以正常添加
            if current_len + word_len <= max_length:
                current_words.append(word)
                current_labels.append(label)
                current_len += word_len
                pointer += 1
            else:
                # 策略1：优先在标点符号处切分
                split_pos = None
                look_back = min(len(current_words), 10)  # 最多回看10个词

                # 查找最近的标点符号
                for i in range(len(current_words) - 1, max(-1, len(current_words) - look_back - 1), -1):
                    if current_words[i] in {"，", "。", "？", "！", "；"}:
                        split_pos = i + 1
                        break

                # 策略2：次优选择（停用词位置）
                if split_pos is None:
                    for i in range(len(current_words) - 1, max(-1, len(current_words) - look_back - 1), -1):
                        if current_words[i] in stop_words and (
                                current_len - sum(len(w) for w in current_words[i + 1:])) >= min_length:
                            split_pos = i + 1
                            break

                # 策略3：保底选择（满足最小长度）
                if split_pos is None:
                    for i in range(len(current_words) - 1, 0, -1):
                        if (current_len - sum(len(w) for w in current_words[i:])) >= min_length:
                            split_pos = i
                            break

                # 执行切分
                if split_pos is not None and split_pos > 0:
                    # 处理前半部分
                    seg_words = current_words[:split_pos]
                    seg_labels = current_labels[:split_pos]
                    train_texts.append("".join(seg_words))
                    train_labels.append(seg_labels)

                    # 保留后半部分
                    current_words = current_words[split_pos:]
                    current_labels = current_labels[split_pos:]
                    current_len = sum(len(w) for w in current_words)
                else:
                    # 无法找到合适切分点，强制切分
                    train_texts.append("".join(current_words))
                    train_labels.append(current_labels)
                    current_words = []
                    current_labels = []
                    current_len = 0

        # 处理剩余部分
        if current_words:
            train_texts.append("".join(current_words))
            train_labels.append(current_labels)

        return train_texts, train_labels

    # 标签映射到id
    def map_tags_to_ids(self, tags2):
        """
        将标签列表中的字符标签映射为整数ID。

        参数:
            tags2 (list of list of str): 包含字符标签的列表。
            list_id (dict): 字符标签到整数ID的映射字典。

        返回:
            mapped_tags (list of list of int): 包含整数ID的列表。
        """
        # 从0开始映射表：
        # labels = [
        #     "O", "B-NR", "E-NR", "M-NR", "S-T", "S-NG", "B-NS", "E-NS", "S-NR",
        #     "B-T", "M-T", "E-T", "S-NS", "S-NO", "B-NO", "M-NO", "E-NO", "B-NG",
        #     "E-NG", "B-NB", "E-NB", "M-NS", "S-NB", "M-NB", "M-NG"
        # ]
        # list_id = {label: idx for idx, label in enumerate(labels)}
        # 从1开始映射表：
        labels = [
            "O", "B-NR", "E-NR", "M-NR", "S-T", "S-NG", "B-NS", "E-NS", "S-NR",
            "B-T", "M-T", "E-T", "S-NS", "S-NO", "B-NO", "M-NO", "E-NO", "B-NG",
            "E-NG", "B-NB", "E-NB", "M-NS", "S-NB", "M-NB", "M-NG"
        ]
        # 创建从1开始的映射
        list_id = {label: idx + 1 for idx, label in enumerate(labels)}
        # 创建一个新的列表来存储映射后的结果
        mapped_tags = []

        # 遍历每一个标签列表
        for tag_list in tags2:
            # 对于每个标签列表，创建一个临时列表来保存转换后的ID,并且前后添加[CLS] 和 [SEP] 的标签为 "O"的ID
            temp_mapped = [list_id["O"]] + [list_id[tag] for tag in tag_list] + [list_id["O"]]
            # 将转换后的ID列表添加到最终的结果列表中
            mapped_tags.append(temp_mapped)

        return mapped_tags

    # 使用模型分词器进行分词,返回：
    def tokenizer(self, tokenizer, texts):
        texts_id = []
        token_type_ids = []
        attention_mask = []
        for item in texts:
            inputs = tokenizer(item, return_tensors="pt")
            texts_id.append(inputs["input_ids"])
            token_type_ids.append(inputs["token_type_ids"])
            attention_mask.append(inputs["attention_mask"])

        return texts_id, token_type_ids, attention_mask

    # 数据预处理总步骤：
    def process_data(self, file_path, tokenizer):
        # 读取数据
        lines = self.read_conll_file(file_path)
        # 获取固定长度的文本与标签：
        texts, tags = self.hybrid_split(lines)
        # 标签转换成数字表示：
        label_ids = self.map_tags_to_ids(tags)
        # 获取文本tokenizer之后的结果
        texts_id, token_type_ids, attention_mask = self.tokenizer(tokenizer, texts)
        max_length = self.max_length
        for i in range(len(label_ids)):
            # 需要填充的长度：
            padding_length = max_length - len(label_ids[i])

            # texts_id填充：torch【1，len(label_ids[i])]  ——> torch[1,max_length]
            new_elements = torch.tensor([[tokenizer.pad_token_id] * padding_length])
            texts_id[i] = torch.cat((texts_id[i], new_elements), dim=1)

            # attention_mask填充: torch【1，len(label_ids[i])]  ——> torch[1,max_length]
            new_elements2 = torch.tensor([[tokenizer.pad_token_id] * padding_length])
            attention_mask[i] = torch.cat((attention_mask[i], new_elements2), dim=1)

            # label_ids填充：list[len(label_ids[i])]  ——> list[max_length]
            label_ids[i] += [0] * padding_length
            # label_ids转换成向量形式: list[max_length] ——> torch[1,max_length]
            label_ids[i] = torch.tensor([label_ids[i]], dtype=torch.int64)

        self.texts_id = texts_id
        self.texts = texts
        self.attention_mask = attention_mask
        self.label_ids = label_ids
        self.labels = tags

        return texts_id, attention_mask, label_ids, texts, tags

    def __len__(self):
        return len(self.texts)

    # 获取文本与标签
    def __getitem__(self, idx):
        texts_id = self.texts_id[idx]
        label_ids = self.label_ids[idx]
        # 创建 attention mask
        attention_mask = self.attention_mask[idx]

        return {
            "input_ids": texts_id,
            "attention_mask": attention_mask,
            "labels": label_ids
        }

#训练类
class train_class:
    def __init__(self,model,tokenizer,file_path):
        self.model=model
        self.tokenizer=tokenizer
        self.file_path=file_path

   #训练函数：
    def train(self,max_length,batch_size,epoch,lr):
        model=self.model
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
        train_dataset = dataprocess(max_length)
        train_dataset.process_data(self.file_path, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        # 设置优化器
        optimizer = AdamW(model.parameters(), lr=lr)

        # 训练循环
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cuda:0")
        model.to(device)

        epochs = epoch
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):  # 使用enumerate获取batch的索引
                input_ids = torch.stack(batch["input_ids"]).to(device)
                attention_mask = torch.stack(batch["attention_mask"]).to(device)
                labels = torch.stack(batch["labels"]).to(device)

                # 移除大小为1的维度
                input_ids = input_ids.squeeze(1).to(device)
                attention_mask = attention_mask.squeeze(1).to(device)
                labels = labels.squeeze(1).to(device)
                # print(labels.shape)


                # 前向计算获得结果：
                ner_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = ner_logits.logits
                # print(logits.shape)

                # 创建损失函数实例，指定 ignore_index=0 来忽略 padding 部分
                loss_function = nn.CrossEntropyLoss(ignore_index=0)

                # 计算损失前调整logits和labels的形状
                logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, num_labels]
                labels = labels.view(-1)  # [batch_size * seq_len]
                # # 计算损失
                loss = loss_function(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # 打印当前batch的信息
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Batch Loss: {loss.item()}")

            average_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")
            self.model=model

    def save_model(self,output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"Model and tokenizer saved to {output_dir}")



















