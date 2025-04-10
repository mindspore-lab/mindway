from transformers import AutoModelForCausalLM, AutoTokenizer
from mindway.transformers import AutoModelForCausalLM as MSAutoModelForCausalLM

checkpoint = "Salesforce/codegen-350M-mono"
model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text = "def hello_world():"

completion = model.generate(**tokenizer(text, return_tensors="pt"))

print(f"torch result: {tokenizer.decode(completion[0])}")

ms_model = MSAutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

completion = ms_model.generate(**tokenizer(text, return_tensors="np"))

print(f"ms result: {tokenizer.decode(completion[0])}")
