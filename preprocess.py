from datasets import load_dataset
import torch
from transformers import BertTokenizer
import logging
import os
import re
from torch.utils.data import Dataset
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    """
    简单的文本清理
    """
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"\'s", " 's", text)
    text = re.sub(r"\'ve", " 've", text)
    text = re.sub(r"\'t", " 't", text)
    text = re.sub(r"\'m", " 'm", text)
    text = re.sub(r"\'ll", " 'll", text)
    text = re.sub(r"\'d", " 'd", text)
    text = re.sub(r"\'re", " 're", text)
    return text

class CNNDailyMailDataset(Dataset):
    """
    创建CNN-DailyMail数据集的封装类
    """
    def __init__(self, articles, summaries, tokenizer, max_source_length=512, max_target_length=128):
        # 数据集长度验证
        if len(articles) != len(summaries):
            raise ValueError("文章数量和摘要数量不匹配")
        
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"使用eos_token作为pad_token: {self.tokenizer.pad_token}")

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = clean_text(self.articles[idx])
        summary = clean_text(self.summaries[idx])

        # 对文章进行编码
        article_encoding = self.tokenizer(
            article,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 对摘要进行编码
        summary_encoding = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 处理标签，将pad_token_id替换为-100以便在损失计算中忽略
        labels = summary_encoding['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': article_encoding['input_ids'].squeeze(0),
            'attention_mask': article_encoding['attention_mask'].squeeze(0),
            'labels': labels
        }

def preprocess_data():
    """
    预处理CNN-DailyMail数据集
    """
    # 加载数据集
    logger.info("加载CNN-DailyMail数据集(版本3.0.0)...")
    try:
        dataset = load_dataset("cnn_dailymail", "3.0.0")
    except Exception as e:
        logger.error(f"加载数据集失败: {str(e)}")
        raise

    # 加载BERT-base-cased tokenizer
    logger.info("加载BERT-base-cased tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # 处理训练集
    logger.info("处理训练集...")
    train_dataset = CNNDailyMailDataset(
        dataset['train']['article'],
        dataset['train']['highlights'],
        tokenizer
    )

    # 处理验证集
    logger.info("处理验证集...")
    val_dataset = CNNDailyMailDataset(
        dataset['validation']['article'],
        dataset['validation']['highlights'],
        tokenizer
    )
    
    # 处理测试集
    logger.info("处理测试集...")
    test_dataset = CNNDailyMailDataset(
        dataset['test']['article'],
        dataset['test']['highlights'],
        tokenizer
    )
    
    # 创建保存目录
    os.makedirs('processed_data', exist_ok=True)
    
    # 保存预处理结果
    logger.info("保存预处理结果...")
    torch.save(train_dataset, 'processed_data/train_dataset.pt')
    torch.save(val_dataset, 'processed_data/val_dataset.pt')
    torch.save(test_dataset, 'processed_data/test_dataset.pt')
    
    # 保存tokenizer
    os.makedirs('processed_data/tokenizer', exist_ok=True)
    tokenizer.save_pretrained('processed_data/tokenizer')
    
    logger.info("数据集预处理完成")

    return train_dataset, val_dataset, test_dataset, tokenizer

def main():
    # 确保保存目录存在
    os.makedirs('processed_data', exist_ok=True)
    
    # 进行预处理
    train_dataset, val_dataset, test_dataset, tokenizer = preprocess_data()
    
    # 打印数据信息
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")
    
    # 打印样例数据
    sample = train_dataset[0]
    logger.info("\n样例数据:")
    logger.info(f"输入文章长度: {len(tokenizer.decode(sample['input_ids']))}")
    logger.info(f"摘要长度: {len(tokenizer.decode(sample['labels'], skip_special_tokens=True))}")


if __name__ == "__main__":
    main()