import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --- Configuration ---
REPO_ID = "nwokikeonyeka/Igbo-Phi3-Bilingual-Chat-v1-merged-Q5_K_M-GGUF"
FILENAME = "igbo-phi3-bilingual-chat-v1-merged-q5_k_m.gguf"

def main():
    print("--- 📥 Checking for model... ---")
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        print(f"--- ✅ Model found at: {model_path} ---")
    except Exception as e:
        print(f"Error downloading model: {e}")
        return

    print("--- 🧠 Loading Brain... (Please wait) ---")
    # n_gpu_layers=-1 tries to move all math to the GPU if available. 
    # If not, it runs on CPU.
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,       # Context window (how much it remembers)
        n_gpu_layers=-1,  # Use GPU if possible
        verbose=False
    )

    print("\n" + "="*50)
    print("🤖 IGBO BILINGUAL CHATBOT")
    print("Commands: Type 'exit' to quit.")
    print("="*50)

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Ka o di! (Goodbye!)")
                break
            
            if not user_input.strip():
                continue

            # --- KEY: Using the Phi-3 Prompt Template ---
            # If we don't use this, the model gets confused.
            prompt = f"<s><|user|>\n{user_input}<|end|>\n<|assistant|>\n"

            print("AI is typing...", end="", flush=True)
            
            # Run Inference
            output = llm(
                prompt,
                max_tokens=512,      # Maximum length of answer
                stop=["<|end|>"],    # Stop generating when done
                echo=False,          # Don't repeat the user's input
                temperature=0.4      # Creativity level (0.1 = robot, 0.8 = poet)
            )

            # Clear the "AI is typing..." line
            print("\r" + " "*20 + "\r", end="")
            
            response = output['choices'][0]['text']
            print(f"Igbo AI: {response}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
