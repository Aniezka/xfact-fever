# When Scale Meets Diversity: Evaluating Language Models on Fine-Grained Multilingual Claim Verification

The link to the paper: https://aclanthology.org/2025.fever-1.5/

## 📄 Overview

This repository contains the **LLMs outputs and evaluation results** from our comprehensive study comparing different language model architectures on multilingual fact verification. We evaluated both small language models (SLMs) and large language models (LLMs) on the challenging X-Fact dataset, which spans 25 languages with seven distinct veracity categories.

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

## 📊 Downloading Model Outputs

The complete model outputs are available as a GitHub release:

### 📥 [Download Model Outputs (Latest Release)](../../releases/latest)

**Quick Download:**
```bash
# Download and extract outputs
wget https://github.com/Aniezka/xfact-fever/releases/latest/download/multilingual-fact-verification-outputs.tar.gz
tar -xzf multilingual-fact-verification-outputs.tar.gz
```

### What's Included

The download contains **model outputs for the three large language models** evaluated in our study. While we evaluated both small language models (XLM-R, mT5) and large language models, we provide outputs only for the LLMs:

```
outputs/
├── benchmarking/
│   └── finetune/                                        # LoRA Fine-tuning Results
│       ├── no_evidence/                                 # Claim-only fine-tuning
│       └── with_evidence/                               # Claim+evidence fine-tuning
│           ├── llama/
│           │   ├── out_finetune_llama_dev.csv
│           │   ├── out_finetune_llama_ood.csv
│           │   ├── out_finetune_llama_test.csv
│           │   └── out_finetune_llama_zeroshot.csv
│           ├── mistral/
│           └── qwen/
└── inference/                                           # Few-shot Prompting Results
    ├── no_evidence/                                     # Claim-only prompting
    │   └── llama/
    │       ├── inference_claim_out_llama_dev.csv
    │       ├── inference_claim_out_llama_ood.csv
    │       ├── inference_claim_out_llama_test.csv
    │       └── inference_claim_out_llama_zeroshot.csv
    ├── mistral/
    ├── qwen/
    └── with_evidence/                                   # Claim+evidence prompting
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
- **Evaluation splits**: In-domain, out-of-domain, zero-shot cross-lingual

### LLM Limitations Discovered
1. **Evidence integration failure**: Sequential processing in decoder-only models hinders balanced evidence evaluation
2. **Class imbalance sensitivity**: LLMs show stronger bias toward frequent categories
3. **Cross-lingual brittleness**: Steeper performance drops on unseen languages


## 📋 Citation

If you use these outputs in your research, please cite our paper:
```
@inproceedings{shcharbakova-etal-2025-scale,
    title = "When Scale Meets Diversity: Evaluating Language Models on Fine-Grained Multilingual Claim Verification",
    author = "Shcharbakova, Hanna  and
      Anikina, Tatiana  and
      Skachkova, Natalia  and
      Genabith, Josef Van",
    editor = "Akhtar, Mubashara  and
      Aly, Rami  and
      Christodoulopoulos, Christos  and
      Cocarascu, Oana  and
      Guo, Zhijiang  and
      Mittal, Arpit  and
      Schlichtkrull, Michael  and
      Thorne, James  and
      Vlachos, Andreas",
    booktitle = "Proceedings of the Eighth Fact Extraction and VERification Workshop (FEVER)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.fever-1.5/",
    pages = "69--84",
    ISBN = "978-1-959429-53-1",
    abstract = "The rapid spread of multilingual misinformation requires robust automated fact verification systems capable of handling fine-grained veracity assessments across diverse languages. While large language models have shown remarkable capabilities across many NLP tasks, their effectiveness for multilingual claim verification with nuanced classification schemes remains understudied. We conduct a comprehensive evaluation of five state-of-the-art language models on the X-Fact dataset, which spans 25 languages with seven distinct veracity categories. Our experiments compare small language models (encoder-based XLM-R and mT5) with recent decoder-only LLMs (Llama 3.1, Qwen 2.5, Mistral Nemo) using both prompting and fine-tuning approaches. Surprisingly, we find that XLM-R (270M parameters) substantially outperforms all tested LLMs (7-12B parameters), achieving 57.7{\%} macro-F1 compared to the best LLM performance of 16.9{\%}. This represents a 15.8{\%} improvement over the previous state-of-the-art (41.9{\%}), establishing new performance benchmarks for multilingual fact verification. Our analysis reveals problematic patterns in LLM behavior, including systematic difficulties in leveraging evidence and pronounced biases toward frequent categories in imbalanced data settings. These findings suggest that for fine-grained multilingual fact verification, smaller specialized models may be more effective than general-purpose large models, with important implications for practical deployment of fact-checking systems."
}
```

## 🔗 Related Resources

- **X-Fact Paper**: https://aclanthology.org/2021.acl-short.86.pdf

## Contact

For any questions or collaboration:
- **Hanna Shcharbakova** - Saarland University, aniezka.sherbakova@gmail.com
