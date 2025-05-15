from datasets import load_dataset
import torch
from transformers import (
    BertTokenizer,
    EncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import numpy as np
from rouge_score import rouge_scorer
import logging
import os
import re
from torch.utils.data import Dataset
from pathlib import Path

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    """
    clean the text
    """
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"\'s", " 's", text)
    text = re.sub(r"\'ve", " 've", text)
    return text

class CNNDailyMailDataset(Dataset):
    """
    create a dataset for the CNN-DailyMail dataset
    """
    def __init__(self, articles, summaries, tokenizer, max_source_length=512, max_target_length=128):
        # dataset length validation
        if len(articles) != len(summaries):
            raise ValueError("The number of article and number of summaries does not match")
        
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Using eos_token as pad_token: {self.tokenizer.pad_token}")

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = clean_text(self.articles[idx])
        summary = clean_text(self.summaries[idx])

        # encode article
        article_encoding = self.tokenizer(
            article,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # encode highlight
        summary_encoding = self.tokenizer(
            text_target=summary,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = summary_encoding['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': article_encoding['input_ids'].squeeze(0),
            'attention_mask': article_encoding['attention_mask'].squeeze(0),
            'labels': labels
        }

def preprocess_data():
    """
    预处理数据
    """
    # 设置输出目录
    output_dir = "E:/data_mining_output/processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # load data
    logger.info("加载CNN-DailyMail数据集...")
    try:
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    except Exception as e:
        logger.error(f"加载数据集失败: {str(e)}")
        raise

    # loading the bert tokenizer
    logger.info("加载BERT-cased tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # process the train dataset
    train_dataset = CNNDailyMailDataset(dataset['train']['article'],
                                      dataset['train']['highlights'],
                                      tokenizer
                                      )

    # process the validation dataset
    val_dataset = CNNDailyMailDataset(dataset['validation']['article'],
                                    dataset['validation']['highlights'],
                                    tokenizer
                                    )
    
    # process the test dataset
    test_dataset = CNNDailyMailDataset(dataset['test']['article'],
                                    dataset['test']['highlights'],
                                    tokenizer
                                    )
    
    # Save the preprocess results
    logger.info("保存预处理结果...")
    torch.save(train_dataset, os.path.join(output_dir, 'train_dataset.pt'))
    torch.save(val_dataset, os.path.join(output_dir, 'val_dataset.pt'))
    torch.save(test_dataset, os.path.join(output_dir, 'test_dataset.pt'))
    
    # save the tokenizer
    tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))
    
    logger.info("数据集预处理成功")

    return train_dataset, val_dataset, test_dataset, tokenizer

def main():
    # 创建输出目录
    os.makedirs("E:/data_mining_output/processed_data", exist_ok=True)
    
    # 进行预处理
    train_dataset, val_dataset, test_dataset, tokenizer = preprocess_data()
    
    # 打印数据信息
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")
    
    # 打印样本数据
    sample = train_dataset[0]
    logger.info("\n样本数据:")
    logger.info(f"输入文章长度: {len(tokenizer.decode(sample['input_ids']))}")
    logger.info(f"摘要长度: {len(tokenizer.decode(sample['labels']))}")


if __name__ == "__main__":
    main()
