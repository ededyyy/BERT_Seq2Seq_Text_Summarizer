import torch
from transformers import (
    BertTokenizer,
    BertConfig,
    EncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    TrainerCallback
)
from rouge_score import rouge_scorer
import logging
import numpy as np
from pathlib import Path
import os
import random
from torch.utils.data import Dataset
from preprocess import CNNDailyMailDataset, clean_text

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """
    load the preprocessed results
    """
    logger.info("Loading the preprocessed data...")
    data_path = Path("processed_data")
    
    # 加载数据集
    train_dataset = torch.load(data_path / 'train_dataset.pt')
    val_dataset = torch.load(data_path / 'val_dataset.pt')
    test_dataset = torch.load(data_path / 'test_dataset.pt')
    tokenizer = BertTokenizer.from_pretrained(data_path / 'tokenizer')
    
    return train_dataset, val_dataset, test_dataset, tokenizer

def bert_seq2seq_model(tokenizer):
    """
    create the BERT-Seq2Seq model
    use bert-base-cased as encoder and decoder
    """
    # create decoder config
    decoder_config = BertConfig.from_pretrained("../bert/models/bert-base-cased")
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    
    # create model using pretrained model names
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path="../bert/models/bert-base-cased",
        decoder_pretrained_model_name_or_path="../bert/models/bert-base-cased",
        decoder_config=decoder_config
    )
    
    # 设置特殊token
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.cls_token_id
    
    # 在decoder配置中也设置相同的token id
    model.config.decoder.decoder_start_token_id = tokenizer.cls_token_id
    model.config.decoder.bos_token_id = tokenizer.cls_token_id
    model.config.decoder.eos_token_id = tokenizer.sep_token_id
    model.config.decoder.pad_token_id = tokenizer.pad_token_id
    
    # 确保模型知道这些token id
    model.decoder.config.decoder_start_token_id = tokenizer.cls_token_id
    model.decoder.config.bos_token_id = tokenizer.cls_token_id
    model.decoder.config.eos_token_id = tokenizer.sep_token_id
    model.decoder.config.pad_token_id = tokenizer.pad_token_id
    
    def custom_loss(logits, labels):
        """
        loss function
        """
        # create mask, ignore padding token
        mask = (labels != -100).float()
        
        # calculate the cross entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        
        loss = loss_fct(logits, labels)
        loss = loss * mask.view(-1)
        
        return loss.sum() / mask.sum()
    
    # add the custom loss function to the model
    model.custom_loss = custom_loss
    
    return model

class Seq2SeqCallback(TrainerCallback):
    """
    callback for the Seq2Seq model
    calculate ROUGE scores during training
    """
    def __init__(self, tokenizer, eval_dataset, sample_size=3, eval_size=100, 
                 num_beams=4, length_penalty=2.0, no_repeat_ngram_size=3):
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.sample_size = min(sample_size, len(eval_dataset))
        self.eval_size = min(eval_size, len(eval_dataset))
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # fix the random seed
        random.seed(42)
        self.example_indices = random.sample(range(len(eval_dataset)), self.sample_size)

    def generate_with_beam_search(self, model, input_ids, attention_mask):
        """
        generate with beam search
        """
        # 使用beam search生成摘要
        # 不再使用logits_processor，改为直接使用model.generate的内置参数
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            min_length=5,  # replace MinLengthLogitsProcessor
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            early_stopping=True,
            no_repeat_ngram_size=self.no_repeat_ngram_size,  # 已经包含NoRepeatNGramLogitsProcessor的功能
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.sep_token_id,
            bos_token_id=self.tokenizer.cls_token_id,
            decoder_start_token_id=self.tokenizer.cls_token_id,
        )
        return outputs

    def on_epoch_end(self, args, state, control, model, **kwargs):
        """
        Called at the end of each epoch.
        """
        model.eval()
        device = model.device
        
        print("\n" + "="*60)
        print(f"Beam Search Evaluation (epoch {state.epoch})")
        print(f"Parameters: beams={self.num_beams}, length_penalty={self.length_penalty}, no_repeat_ngram={self.no_repeat_ngram_size}")
        
        # show the generated examples
        print("\nGenerated Examples:")
        for idx in self.example_indices:
            sample = self.eval_dataset[idx]
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            
            # generate with beam search
            outputs = self.generate_with_beam_search(model, input_ids, attention_mask)
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            reference = self.tokenizer.decode(
                [t for t in sample['labels'] if t != -100], 
                skip_special_tokens=True
            )
            
            print(f"\nInput Text: {self.tokenizer.decode(input_ids[0], skip_special_tokens=True)[:150]}...")
            print(f"Reference Summary: {reference}")
            print(f"Generated Summary: {generated}")

        # calculate the ROUGE scores
        print(f"\nCalculating ROUGE scores (based on {self.eval_size} samples)...")
        rouge_scores = []
        eval_indices = random.sample(range(len(self.eval_dataset)), self.eval_size)
        
        for idx in eval_indices:
            sample = self.eval_dataset[idx]
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            
            # 使用相同的生成参数
            outputs = self.generate_with_beam_search(model, input_ids, attention_mask)
            
            scores = self.scorer.score(
                self.tokenizer.decode([t for t in sample['labels'] if t != -100], skip_special_tokens=True),
                self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            )
            rouge_scores.append({
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            })
        
        # 计算平均分数
        avg_rouge1 = np.mean([s['rouge1'] for s in rouge_scores])
        avg_rouge2 = np.mean([s['rouge2'] for s in rouge_scores])
        avg_rougeL = np.mean([s['rougeL'] for s in rouge_scores])
        
        print("\nROUGE Scores:")
        print(f"ROUGE-1: {avg_rouge1:.4f}")
        print(f"ROUGE-2: {avg_rouge2:.4f}")
        print(f"ROUGE-L: {avg_rougeL:.4f}")
        print("="*60 + "\n")
        
        model.train()

def evaluate_model(pred, tokenizer):
    """
    计算ROUGE分数
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    predictions = pred.predictions
    labels = pred.label_ids
    
    # 如果predictions是logits，需要进行argmax
    if len(predictions.shape) == 3:
        predictions = np.argmax(predictions, axis=-1)
    
    # 解码预测和标签
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # 替换-100为pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 计算ROUGE分数
    scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }
    
    for pred, label in zip(decoded_preds, decoded_labels):
        score = scorer.score(label, pred)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)
    
    # 计算平均分数
    return {
        'rouge1': np.mean(scores['rouge1']),
        'rouge2': np.mean(scores['rouge2']),
        'rougeL': np.mean(scores['rougeL'])
    }

def main():
    # set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # load the data
    train_dataset, val_dataset, test_dataset, tokenizer = load_data()
    logger.info(f"Train data size: {len(train_dataset)}")
    logger.info(f"Val data size: {len(val_dataset)}")
    logger.info(f"Test data size: {len(test_dataset)}")
    
    # initialize the model
    logger.info("Initializing the BERT-Seq2Seq model...")
    model = bert_seq2seq_model(tokenizer)
    model.to(device)
    
    # set up the training parameters
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        eval_strategy="no",
        #eval_steps=50,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,
        max_steps=100,
        # num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,          # 每50步保存一次
        load_best_model_at_end=False,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        # radient_accumulation_steps=4,
        # generation_max_length=128,
        # generation_num_beams=4,
    )
    
    # create the data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    
    # create the callback with custom parameters
    callback = Seq2SeqCallback(
        tokenizer=tokenizer,
        eval_dataset=val_dataset,
        sample_size=3,          # 展示3个样本
        eval_size=100,          # 评估100个样本
        num_beams=4,            # 使用4个beam
        length_penalty=2.0,     # 长度惩罚系数
        no_repeat_ngram_size=3  # 避免3-gram重复
    )
    
    # create the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: evaluate_model(x, tokenizer),
        callbacks=[callback],
    )
    
    # start training
    logger.info("Start training...")
    trainer.train()
    
    # save the best model
    logger.info("Saving the best model...")
    trainer.save_model("./best_model")
    
    # 确保在评估前正确设置特殊token
    logger.info("确保特殊token设置正确...")
    # 显式设置模型配置中的必要token ID
    trainer.model.config.decoder_start_token_id = tokenizer.cls_token_id
    trainer.model.config.bos_token_id = tokenizer.cls_token_id
    trainer.model.config.eos_token_id = tokenizer.sep_token_id
    trainer.model.config.pad_token_id = tokenizer.pad_token_id
    
    # evaluate on test set with beam search
    logger.info("Evaluating on test set with beam search...")
    # 调整生成参数，确保包含必要的token IDs
    generation_config = {
        "max_length": 128,
        "num_beams": 4,
        "decoder_start_token_id": tokenizer.cls_token_id,
        "bos_token_id": tokenizer.cls_token_id,
        "eos_token_id": tokenizer.sep_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    # 控制测试集大小，只取前N个样本进行评估
    test_subset_size = 100  # 设置为None使用全部测试集
    if test_subset_size and test_subset_size < len(test_dataset):
        logger.info(f"使用测试集的子集进行评估 (大小: {test_subset_size}，总样本数量: {len(test_dataset)})")
        test_subset = torch.utils.data.Subset(test_dataset, range(test_subset_size))
        test_results = trainer.evaluate(test_subset, **generation_config)
    else:
        test_results = trainer.evaluate(test_dataset, **generation_config)
    
    logger.info("测试集评估结果:")
    for key, value in test_results.items():
        logger.info(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()