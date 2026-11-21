| Model | Accuracy | OOS recall | Size (MB) | Latency (ms) |
|---|---|---|---|---|
| tf-idf + LogReg | 77.3% | 15.8% | 30.5 | 23.2 ± 1.4 |
| word2vec + FFN (SGD) | 78.3% | 31.6% | 2.4 | 0.2 ± 0.0 |
| BERT-base fine-tuned (fp32) | 87.2% | 46.1% | 418.2 | 103.8 ± 18.2 |
| DistilBERT-KD (fp32) | 86.5% | 44.4% | 255.9 | 52.6 ± 8.0 |
| DistilBERT-KD (INT8) | 86.7% | 48.1% | 132.4 | 14.5 ± 2.9 |
