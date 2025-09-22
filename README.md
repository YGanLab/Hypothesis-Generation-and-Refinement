# Hypothesis-Generation-and-Refinement

This repository contains the complete pipeline and codebase for the article:

**"Iterative Hypothesis Generation in Chemistry Using Memory-Augmented Large Language Models"**

## 🧬 Overview

This repository provides a **Retrieval-Augmented Generation (RAG)** pipeline for **scientific hypothesis generation, refinement, and evaluation**.  
It integrates **PubMed, Semantic Scholar, and CrossRef retrieval** with **LLM-based reasoning** to produce, refine, and score research hypotheses across multiple rounds.


## 📁 Project Structure

```
main.py                         # Entry point for running the hypothesis generation pipeline
rag.py                          # Retrieval-Augmented Generation (RAG) utilities and LLM interfaces
retrieval.py                    # Retrieval tools for PubMed, Semantic Scholar, and CrossRef
config.yaml                     # User-defined configuration for APIs, prompts, and dataset paths
utilities.py                    # Helper functions (dataset loading, parsing, evaluation)
```

## 🚀 Features

- **Keyword Extraction & Query Generation**  
  Uses LLMs to extract keywords and build flexible PubMed queries.

- **Multi-Source Retrieval**  
  Fetches related articles from PubMed, Semantic Scholar, and CrossRef.

- **Hypothesis Generation & Refinement**  
  Iteratively generates and refines hypotheses based on background questions, retrieved papers, and LLM feedback.

- **LLM-Based Scoring**  
  Hypotheses are evaluated on **novelty, specificity, plausibility, validity, and significance**, inspired by peer-review criteria.

- **Round-Based Iteration**  
  Supports multi-round refinement with contextual memory to encourage progress across iterations.

---


## ⚙️ Dependencies

- python>=3.9
- torch, transformers, sentence-transformers
- openai, anthropic
- numpy, pandas, tqdm
- pyyaml, requests, lxml

Ensure you have API keys set in config.yaml for OpenAI and/or Anthropic, and optionally PubMed, Semantic Scholar, and CrossRef.

## 📊 Usage

### 1. Configure the Pipeline

- Edit config.yaml with:
- API keys (OpenAI, Anthropic, PubMed, etc.)
- Dataset path
- Number of refinement rounds
- Prompt templates

### 2. Run the Pipeline

```bash
python main.py
```
This will:
- Load the dataset (background questions, surveys, and ground-truth hypotheses).
- Generate PubMed/Semantic Scholar/CrossRef queries.
- Retrieve relevant articles.
- Generate hypotheses, refine them across multiple rounds, and score each.
- Print per-round average scores and final hypothesis quality summary.


## 📈 Example Workflow

- Input: Background question + short literature survey.
- Retrieval: Fetches related articles and abstracts.
- Hypothesis Generation: Produces 8–10 candidate hypotheses.
- Refinement: Iteratively improves hypotheses using feedback and new inspirations.
- Evaluation: Scores novelty, specificity, plausibility, and selects the best final hypothesis.


## 📝 Citation

If you use this code, please cite our work:

@misc{hypopipeline2025,
  author = {Mahdi Babaei},
  title  = {Iterative Hypothesis Generation in Chemistry Using Memory-Augmented Large Language Models},
  year   = {2025},
  url    = {https://github.com/MahdiBabaei96/hypothesis-pipeline}
}


## 📬 Contact

For questions or collaborations, please contact Mahdi Babaei at [mbabaei1@stevens.edu].

