# Which RL Loss Should a Small Model Use? (0.5B RLVR)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/0.5B_RLVR_Demo.ipynb)
[![Model Weights](https://img.shields.io/badge/Weights-GitHub_Releases-blue)](#model-checkpoints)

This repository contains the code, models, and evaluation logs for our study: **Testing GRPO, DAPO, and Dr. GRPO at 0.5B**. We analyze how loss normalization choices and standard deviation scaling impact Reinforcement Learning with Verifiable Rewards (RLVR) in sub-1B models.

## 🚀 Quickstart: 1-Click Demonstration
Want to see the training loop in action without setting up a local environment? Click the **Open in Colab** badge at the top of this README to run a demonstration using our custom TRL patch on a subset of GSM8K.

## 📦 Model Checkpoints
We have released our best-performing Qwen2.5-0.5B-Instruct checkpoint trained via our optimal RL loss configuration. 
* **Download:** [Available in GitHub Releases](https://github.com/YOUR_USERNAME/YOUR_REPO/releases/tag/v1.0-weights)
* **Usage:** You can load these weights locally using Hugging Face `transformers`.

## 🛠️ Installation & Setup
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO.git](https://github.com/YOUR_USERNAME/YOUR_REPO.git)
cd YOUR_REPO
pip install -r requirements.txt
