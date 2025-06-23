# When Scale Meets Diversity: Evaluating Language Models on Fine-Grained Multilingual Claim Verification

## 📄 Overview

This repository contains the **model outputs and evaluation results** from our comprehensive study comparing different language model architectures on multilingual fact verification. We evaluated both small language models (XLM-R base, mT5 base) and large language models (Qwen 2.5 7B, llama 3.1 8B, Mistral Nemo 12B) on the challenging X-Fact dataset, which spans 25 languages with seven distinct veracity categories.

> **Key Finding**: XLM-R (270M parameters) substantially outperforms all tested LLMs (7-12B parameters), achieving **57.7% macro-F1** compared to the best LLM performance of **16.9%** - a **15.8% improvement** over previous state-of-the-art!

## 🎯 Key Results

### Performance Summary
| Model | Parameters | Architecture | Test Macro-F1 | OOD Macro-F1 | Zero-shot Macro-F1 |
|-------|------------|--------------|---------------|--------------|-------------------|
| **XLM-R** (full) | 270M | Encoder-only | **57.7%** | **47.6%** | **43.2%** |
| XLM-R (frozen) | 270M | Encoder-only | 51.4% | 40.8% | 41.3% |
| mT5 | 580M | Encoder-decoder | 47.6% | 22.2% | 19.2% |
| Qwen 2.5 | 7B | Decoder-only | 16.9% | 11.1% | 11.7% |
| Mistral Nemo | 12B | Decoder-only | 14.8% | 16.1% | 15.1% |
| Llama 3.1 | 8B | Decoder-only | 15.5% | 13.5% | 12.1% |

### Surprising Findings
- 🔍 **Evidence doesn't always help**: LLMs often performed worse when provided with claim+evidence pairs compared to claims alone
- 📊 **Scale paradox**: Smaller specialized models outperform much larger general-purpose models
- 🌐 **Cross-lingual gaps**: All models show performance degradation on out-of-domain and zero-shot language transfer

## 📊 What's Included

This repository contains **model outputs for the three large language models** evaluated in our study. While we evaluated both small language models (XLM-R, mT5) and large language models, we provide outputs only for the LLMs:

```
outputs/
├── benchmarking/
│   └── finetune/
│       ├── no_evidence/
│       └── with_evidence/
│           ├── llama/
│           │   ├── out_finetune_llama_dev.csv
│           │   ├── out_finetune_llama_ood.csv
│           │   ├── out_finetune_llama_test.csv
│           │   └── out_finetune_llama_zeroshot.csv
│           ├── mistral/
│           └── qwen/
└── inference/
    ├── no_evidence/
    │   └── llama/
    │       ├── inference_claim_out_llama_dev.csv
    │       ├── inference_claim_out_llama_ood.csv
    │       ├── inference_claim_out_llama_test.csv
    │       └── inference_claim_out_llama_zeroshot.csv
    ├── llama_new_prompt_1/
    ├── llama_new_prompt_3/
    ├── llama_old_prompt/
    ├── mistral/
    ├── qwen/
    └── with_evidence/
```

### Output Format
Each CSV file contains the following columns:
- `claim_evidence`: The input text (claim only or claim + evidence)
- `predicted_text`: Raw model output text
- `parsed_predicted_label`: Extracted prediction label from model output
- `true_label`: Ground truth veracity category

### File Naming Convention
- `*_dev.csv`: Development set predictions
- `*_test.csv`: Test set predictions (in-domain)
- `*_ood.csv`: Out-of-domain test set predictions
- `*_zeroshot.csv`: Zero-shot cross-lingual test set predictions

## 🔬 Experimental Setup

### Models Evaluated
- **Small Language Models**:
  - XLM-R base (270M) - encoder-only, 100 languages
  - mT5 base (580M) - encoder-decoder, 101 languages

- **Large Language Models**:
  - Llama 3.1 8B - decoder-only, 8 languages
  - Qwen 2.5 7B - decoder-only, 29 languages  
  - Mistral Nemo 12B - decoder-only, 11 languages

### Training Approaches
- **SLMs**: Full fine-tuning & classification head fine-tuning
- **LLMs**: Few-shot prompting (7-shot) & LoRA parameter-efficient fine-tuning

### Dataset: X-Fact
- **Languages**: 25 languages across 11 language families
- **Claims**: 31,189 total claims
- **Categories**: 7 fine-grained veracity labels
  - `true`, `mostly_true`, `partly_true`, `mostly_false`, `false`, `complicated`, `other`
- **Evaluation splits**: In-domain, out-of-domain, zero-shot (cross-lingual)

### LLM Limitations Discovered
1. **Evidence integration failure**: Sequential processing in decoder-only models hinders balanced evidence evaluation
2. **Class imbalance sensitivity**: LLMs show stronger bias toward frequent categories
3. **Cross-lingual challenges**: Steeper performance drops on unseen languages

## 📋 Citation

If you use these outputs in your research, please cite our paper:

TBD

## 🔗 Related Resources

- **X-Fact Paper**: https://aclanthology.org/2021.acl-short.86.pdf

## Contact

For questions about the outputs or methodology:
**Hanna Shcharbakova** - Saarland University, aniezka.sherbakova@gmail.com
