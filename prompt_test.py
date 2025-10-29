from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Your Hugging Face token
hf_token = "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"

# Load tokenizer and model directly using the token
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it", use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-12b-it",
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=hf_token
)

prompt = "Translate the following English sentence to Italian: 'Hello, how are you?'"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
