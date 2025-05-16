import torch
from transformers import BertTokenizer, EncoderDecoderModel
from rouge_score import rouge_scorer
import logging
from pathlib import Path
import numpy as np
import os
import random
from preprocess import CNNDailyMailDataset

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_data():
    """
    加载模型和测试数据
    """
    logger.info("加载模型和测试数据...")
    
    # 确保模型路径存在
    model_path = Path("./best_model")
    if not model_path.exists():
        logger.error(f"模型路径不存在: {model_path}")
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    # 加载模型和tokenizer
    model = EncoderDecoderModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # 加载测试数据
    data_path = Path("processed_data")
    if not (data_path / 'test_dataset.pt').exists():
        logger.error(f"测试数据不存在: {data_path / 'test_dataset.pt'}")
        raise FileNotFoundError(f"测试数据不存在: {data_path / 'test_dataset.pt'}")
    
    test_dataset = torch.load(data_path / 'test_dataset.pt')
    
    return model, tokenizer, test_dataset

def evaluate_model(model, tokenizer, test_dataset, num_samples=200, beam_size=4):
    """
    在测试集上评估模型
    使用beam search生成摘要
    """
    logger.info(f"开始评估，使用{num_samples}个样本，beam size={beam_size}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 创建Rouge评分器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }
    
    # 随机选择样本进行评估
    random.seed(42)  # 确保结果可重现
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    for i, idx in enumerate(indices):
        if i % 10 == 0:
            logger.info(f"处理样本 {i+1}/{len(indices)}")
            
        sample = test_dataset[idx]
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        
        # 生成摘要
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=beam_size,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.sep_token_id,
            bos_token_id=tokenizer.cls_token_id,
            decoder_start_token_id=tokenizer.cls_token_id,
        )
        
        # 解码预测和参考文本
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reference = tokenizer.decode(
            [t for t in sample['labels'] if t != -100],
            skip_special_tokens=True
        )
        
        # 计算ROUGE分数
        score = scorer.score(reference, prediction)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)
        
        # 打印一些示例
        if i < 5:  # 展示前5个样本
            print(f"\n示例 {i+1}:")
            article_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print(f"输入文本: {article_text[:150]}...")
            print(f"参考摘要: {reference}")
            print(f"生成摘要: {prediction}")
            print(f"ROUGE-1: {score['rouge1'].fmeasure:.4f}")
            print(f"ROUGE-2: {score['rouge2'].fmeasure:.4f}")
            print(f"ROUGE-L: {score['rougeL'].fmeasure:.4f}")
    
    # 计算平均分数
    avg_scores = {
        'rouge1': np.mean(scores['rouge1']),
        'rouge2': np.mean(scores['rouge2']),
        'rougeL': np.mean(scores['rougeL'])
    }
    
    print("\n评估结果:")
    print(f"ROUGE-1: {avg_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {avg_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {avg_scores['rougeL']:.4f}")
    
    return avg_scores

def analyze_results(model, tokenizer, test_dataset, sample_indices):
    """
    深入分析一些样本的生成结果
    """
    logger.info("进行深入结果分析...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    for i, idx in enumerate(sample_indices):
        sample = test_dataset[idx]
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        
        # 使用不同的beam size
        print(f"\n样本 {i+1} - 不同beam size的对比:")
        article_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        reference = tokenizer.decode(
            [t for t in sample['labels'] if t != -100],
            skip_special_tokens=True
        )
        print(f"输入文本: {article_text[:150]}...")
        print(f"参考摘要: {reference}")
        
        for beam in [1, 4, 8]:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=beam,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.sep_token_id,
                bos_token_id=tokenizer.cls_token_id,
                decoder_start_token_id=tokenizer.cls_token_id,
            )
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Beam={beam} 生成: {prediction}")

def main():
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 加载模型和数据
    try:
        model, tokenizer, test_dataset = load_model_and_data()
    except FileNotFoundError as e:
        logger.error(f"加载模型或数据失败: {e}")
        return
    
    # 评估模型
    scores = evaluate_model(
        model=model, 
        tokenizer=tokenizer, 
        test_dataset=test_dataset, 
        num_samples=200,  # 评估200个样本
        beam_size=4       # beam size=4
    )
    
    # 深入分析一些样本
    random.seed(42)
    sample_indices = random.sample(range(len(test_dataset)), 3)
    analyze_results(model, tokenizer, test_dataset, sample_indices)
    
    # 保存评估结果
    with open('results/evaluation_results.txt', 'w') as f:
        f.write("BERT-base-cased Seq2Seq模型评估结果\n")
        f.write("=================================\n\n")
        f.write(f"ROUGE-1: {scores['rouge1']:.4f}\n")
        f.write(f"ROUGE-2: {scores['rouge2']:.4f}\n")
        f.write(f"ROUGE-L: {scores['rougeL']:.4f}\n")
    
    logger.info(f"评估完成，结果已保存到 results/evaluation_results.txt")

if __name__ == "__main__":
    main()