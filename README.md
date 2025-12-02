# 🤖 Igbo-Phi3-Bilingual-Chat

<div align="center">

![Igbo-AI-Banner](https://img.shields.io/badge/Igbo-AI-green?style=for-the-badge) 
[![Hugging Face GGUF](https://img.shields.io/badge/🤗%20Hugging%20Face-GGUF%20(Local)-orange?style=for-the-badge)](https://huggingface.co/nwokikeonyeka/Igbo-Phi3-Bilingual-Chat-v1-merged-Q5_K_M-GGUF)
[![Hugging Face Merged](https://img.shields.io/badge/🤗%20Hugging%20Face-Master%20Weights%20(Dev)-yellow?style=for-the-badge)](https://huggingface.co/nwokikeonyeka/Igbo-Phi3-Bilingual-Chat-v1-merged)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

</div>

A specialized **Bilingual AI Assistant** trained to converse fluently in **Igbo** and **English**. 

Unlike my previous attempt which was a simple translation model, this AI is a **conversational agent**. It can chat, explain concepts, reason, and define words in both languages while retaining the general intelligence of its base model (Phi-3).

---

## 📥 Download Models

| Version | Best For... | Link |
| :--- | :--- | :--- |
| **GGUF (Q5_K_M)** | **Running locally** on laptops (Mac/Windows/Linux). Fast & Low RAM. | [👉 Download Here](https://huggingface.co/nwokikeonyeka/Igbo-Phi3-Bilingual-Chat-v1-merged-Q5_K_M-GGUF) |
| **Merged (F16)** | **Developers** who want to fine-tune further or use PyTorch. | [👉 Download Here](https://huggingface.co/nwokikeonyeka/Igbo-Phi3-Bilingual-Chat-v1-merged) |

---

## ⚡ Quick Colab Demo

If you don't have a Python environment set up, you can copy-paste this code into a [Google Colab](https://colab.research.google.com/) cell to test the model immediately.

```python
# --- 1. Install Libraries ---
!pip install llama-cpp-python huggingface_hub

# --- 2. Download & Load Model ---
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

REPO_ID = "nwokikeonyeka/Igbo-Phi3-Bilingual-Chat-v1-merged-Q5_K_M-GGUF"
FILENAME = "igbo-phi3-bilingual-chat-v1-merged-q5_k_m.gguf"

print(f"Downloading {FILENAME}...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

print("Loading model...")
llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

# --- 3. Chat Loop ---
print("\n🤖 IGBO CHATBOT READY (Type 'exit' to quit)")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']: break
    
    # Correct Phi-3 Prompt Template
    prompt = f"<s><|user|>\n{user_input}<|end|>\n<|assistant|>\n"
    
    output = llm(prompt, max_tokens=256, stop=["<|end|>"], echo=False)
    print(f"AI: {output['choices'][0]['text']}")
```

---

## 📚 Training Data & Credits

This model was trained on a curated mix of over **700,000 examples** to ensure a balance between language fluency and general logic. Grateful acknowledgment to the creators of these open datasets:

1.  **Fluency (522k pairs):** [ccibeekeoc42/english_to_igbo](https://huggingface.co/datasets/ccibeekeoc42/english_to_igbo)  
    *Primary source for sentence-level translation and grammar.*
2.  **Vocabulary (5k definitions):** [nkowaokwu/ibo-dict](https://huggingface.co/datasets/nkowaokwu/ibo-dict)  
    *Provides deep knowledge of specific Igbo words and definitions.*
3.  **General Memory (200k chats):** [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)  
    *Used to maintain the model's ability to chat, reason, and follow instructions without "forgetting" general knowledge.*

---

## 🚀 Quick Start (Local)

You can run the GGUF model on any computer with Python installed.

### 1. Install Dependencies
```bash
pip install llama-cpp-python huggingface_hub
````

### 2\. Run the Chat Script

Download the `chat.py` file from this repository and run it:

```bash
python chat.py
```

-----

## 🧠 Training Methodology: "The Colab Relay Race"

Training a full LLM on a free Google Colab GPU usually causes timeouts before completion. This project used a **"Relay Race" strategy**:

1.  **Checkpointing:** The training script saves progress every 500 steps to Hugging Face.
2.  **Resuming:** When Colab times out (approx. every 4 hours), a new session is started.
3.  **Relaying:** The script automatically pulls the last checkpoint and resumes training exactly where it stopped.

**Stats:**

  * **Base Model:** Microsoft Phi-3-mini-4k-instruct
  * **Total Steps:** 44,500
  * **Epochs:** 1
  * **Training Time:** \~20 Hours (across multiple sessions)

-----

## 🛠️ Prompt Template

If you use this model in **Ollama**, **LM Studio**, or **Jan.ai**, ensure you use the **Phi-3** prompt format for the best results:

```text
<s><|user|>
{Your Question Here}<|end|>
<|assistant|>
{AI Response Here}<|end|>
```
