# BERT_Seq2Seq_Text_Summarizer

A sequence-to-sequence text summarization project using the CNN/DailyMail dataset and a BERT-base-cased model as both encoder and decoder. This implementation includes preprocessing, training with ROUGE evaluation, and generation using beam search.

## Dataset

This project uses the **CNN/DailyMail v3.0.0** dataset from the `datasets` library. It includes:

- Training Set
- Validation Set
- Test Set

---

## 1. Preprocessing

### Run:

```bash
python preprocess.py
```


## 2. Training

### Run:

```bash
python train.py
```


## 3. Testing

### Run:

```bash
python test.py
```
