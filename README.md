
# LogicReward

**Code and data for ICLR 2026 Paper *[“LogicReward: Incentivizing LLM Reasoning via Step-Wise Logical Supervision ”](https://llm-symbol.github.io/LogicReward/)***

Authors: [**Jundong Xu**](https://aiden0526.github.io/)<sup>1</sup>, [**Hao Fei**](http://haofei.vip/)<sup>1</sup><sup>*</sup>, [**Huichi Zhou**](https://huichizhou.github.io/)<sup>2</sup>, [**Xin Quan**](https://xinquan42.github.io/)<sup>3</sup>, Qijun Huang<sup>4</sup>, [**Shengqiong Wu**](https://sqwu.top/)<sup>1</sup>, [**William Yang wang**](https://sites.cs.ucsb.edu/~william/)<sup>5</sup>, [**Mong-Li Lee**](https://www.comp.nus.edu.sg/cs/people/leeml/)<sup>1</sup>, [**Wynne Hsu**](https://www.comp.nus.edu.sg/cs/people/whsu/)<sup>1</sup>

<sup>1</sup> National University of Singapore, <sup>2</sup> University College London, <sup>3</sup> University of Manchester, <sup>4</sup> University of Melbourne, <sup>5</sup> University of California, Santa Barbara

## 🧠 What is LogicReward?

LogicReward is a reward function that evaluates **unstructured natural language reasoning and provides step-level, symbolically guided rewards**. 


## 🔥 Quick Start

You can directly access the trained models — **LogicReward-Qwen3-8B** and **LogicReward-Llama3.1-8B** — via the following Hugging Face collection:  
https://huggingface.co/collections/Aiden0526/logicreward

You could also use the generated data in `logicreward_data` to directly train the model. Please follow the training and inference details specified in this repo and the paper.


## 📈 Performance

- 🚀 **Consistent gains across benchmarks**:  
  - LogicReward improves **LLaMA-3.1-8B by +11%** and **Qwen-3-8B by +3.2%** on average across **8 logical reasoning and natural language inference benchmarks**. 
  - Using an 8B model, we outperform strong baselines such as **GPT-4o and o4-mini by +11.6% and +2%**.

- 🏆 **Outperforms existing reward signals**:  
  LogicReward demonstrates stronger performance than alternative reward functions, including **confidence-based rewards**, **LLM-as-a-Judge**, and **Process Reward Models (PRMs)**.

- 🌍 **Stronger out-of-distribution generalization**:  
  Models trained with LogicReward generalize better to OOD tasks such as:
  - **Commonsense reasoning** (CommonsenseQA)
  - **Mathematical reasoning** (GSM8K)
  - **Deductive reasoning** (BoardGameQA)

- 🧠 **Faithful reasoning beyond accuracy**:  
  LogicReward improves not only final-task accuracy, but also the **faithfulness, logical consistency, and rigor of intermediate reasoning steps**.


## 🧭 TODO

### ✅ Completed Features

* Full pipeline for **rollout generation**
* **LogicReward-based reward labeling** for reasoning data

### 🚧 Upcoming Improvements

* 🔥 Release **two trained model weights** on Hugging Face
* 🛠️ Improve usability:

  * Support **custom rollout & reward labeling** with **one command**


## 📰 News

* **[2025.12.23]** 🎉 We release the **LogicReward paper on arXiv**!


## 🚀 Quick Start

We will release our trained model weights on **Hugging Face**.

👉 **LogicReward models on Hugging Face:**
[placeholder_website]


## 🛠️ Setup

### 1️⃣ Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```


### 2️⃣ spaCy English Model

```bash
python -m spacy download en_core_web_sm
```


### 3️⃣ Isabelle

Download and unpack **Isabelle 2023**:

```bash
wget https://isabelle.in.tum.de/website-Isabelle2023/dist/Isabelle2023_linux.tar.gz
tar -xzf Isabelle2023_linux.tar.gz --no-same-owner
```

📌 Make note of the path to:

```text
Isabelle2023/bin
```

For example:

```text
/workspace/Isabelle2023/bin
```

You’ll need this path later.


### 4️⃣ vLLM

Install **vLLM** by following the official instructions:
🔗 [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)


## 🔄 Rollout

Navigate to the rollout directory:

```bash
cd rollout
```

### ▶️ Qwen Rollout

```bash
python qwen_rollout.py
```

### ▶️ OpenAI Rollout

For OpenAI-based models:

1. Replace `YOUR_API_KEY` in the helper function with your OpenAI api key.

2. Run the notebook:

```text
openai_rollout.ipynb
```

3. After generating both **Qwen** and **OpenAI** rollouts, run:

```text
prepare_reward_data.ipynb
```

This converts rollout outputs into **reward-ready format**.


## 🧮 LogicReward (Reward Labeling)

### 🔹 Standalone Mode (Direct Execution)

Run LogicReward **without tmux or multiprocessing**:

```bash
python label_reward_batch.py \
    -p <PORT> \              # any free port number
    -i <INPUT_JSONL> \       # rollout input JSONL
    -o <OUTPUT_JSONL> \      # rewarded output JSONL
    -s <SKIP_JSONL> \        # optional: IDs to skip
    -d <RUN_TAG>             # optional: run label (e.g., "standalone")
```


### 🔹 Multi-Thread Mode

To apply **LogicReward** with parallel execution:

1. Open `run_reward.sh` and update:

   * Replace `{ENV_NAME}` with your **conda environment name**
   * Replace `/workspace/Isabelle2023/bin` with your **Isabelle bin path**

2. Open `config.yaml` and:

   * Set your **OpenAI API key**
   * Set your **Hugging Face API token**

3. (Optional) Adjust `N` in `run_reward.sh` to control parallel threads

Then run:

```bash
bash run_reward.sh
```

📂 Output location:

```text
reward_data/Multi-Thread/output/
```

This data will be used for **training dataset construction**.


## 🔧 Refinement

After obtaining reward-labeled data, you can refine the generated reasoning.


### 🔹 Standalone Mode

Run refinement without tmux or multiprocessing:

```bash
python refine_reasoning.py \
    --port <PORT> \                # any free port number
    --input <INPUT_JSONL> \        # rewarded data JSONL
    --output <OUTPUT_JSONL> \      # refined result JSONL
    --processed <SKIP_JSONL> \     # optional: IDs to skip
    --theory-name <THEORY_NAME> \  # e.g., "symb_single"
    --max-workers 1                # ensures standalone refinement
```


### 🔹 Multi-Thread Mode

In `run_refine.sh`:

* Adjust `N` to set the number of parallel threads

Then run:

```bash
bash run_refine.sh
```

📂 Refined outputs will be saved to:

```text
refine_data/Multi-Thread/
```

(See the script for exact subpaths.)


## 📚 Construct SFT and DPO Data

Once you have:

* ✅ **Reward data** (from `run_reward.sh`)
* ✅ **Refinement data** (from `run_refine.sh`)

Run:

```text
construct_training_data.ipynb
```

This notebook:

* Constructs final **training datasets**
* Produces a **DPO dataset**
* Uses `good_answer` as the **SFT target**
* Uses `good_answer` vs. `bad_answer` for **preference learning**


## 🏋️ Training

We use **LLaMA-Factory** for model training.

Please follow the official documentation for both SFT and DPO training:
🔗 [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)





## 📖 Citation

If you use **LogicReward**, the codebase, the data, or the trained models in your research, please **cite our paper**.

```
@article{logicreward2026,
  title   = {LogicReward: Incentivizing LLM Reasoning via Step-Wise Logical Supervision},
  author  = {Jundong Xu, Hao Fei, Huichi Zhou, Xin Quan, Qijun Huang, Shengqiong Wu, William Yang Wang, Mong-Li Lee, Wynne Hsu},
  booktitle = {Proceedings of the International Conference on Learning Representations},
  year    = {2026},
  url = {https://arxiv.org/abs/2512.18196}
}
```

