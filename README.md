# Design of Transformer-Based QA Systems for Logistic Data

This repository contains the code, dataset generation scripts, model configurations, and evaluation utilities for the paper:

**Design of Transformer-Based QA Systems for Logistic Data**  
[ITAT 2025 Conference Submission]  
Authors: Martin Bača, Šimon Horvát  
Affiliation: Pavol Jozef Šafárik University in Košice

---

## 🧠 Overview
This project presents an end-to-end Question Answering (QA) pipeline tailored for the logistics domain. It covers:

- Automatic generation of realistic shipment descriptions from structured JSON
- Synthetic QA pair creation using LLaMA 3.3
- Dataset formatting in SQuADv2-style JSON
- Fine-tuning and evaluation of three transformer-based models:
  - `XtremeDistil l12 h384 Uncased`
  - `XLM-RoBERTa Base`
  - `XLM-RoBERTa Large`

The system handles both **extractive** and **boolean (yes/no)** question types using a unified model architecture with a dual QA head.

---

## 📊 Results

Evaluation on the synthetic dataset shows:

| Model               | F1 (Extractive) | F1 (Boolean) |
|--------------------|-----------------|--------------|
| XtremeDistil       | 83.4%           | 77.5%        |
| XLM-RoBERTa Base   | 93.5%           | 90.1%        |
| XLM-RoBERTa Large  | 95.5%           | 92.1%        |

---

## 📬 Contact

For questions, reach out to:

- martin.baca@student.upjs.sk  
- simon.horvat@upjs.sk

