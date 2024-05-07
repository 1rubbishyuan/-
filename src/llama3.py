from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from string import Template

prompt_template = """
请告诉我接下来这句话表达什么情感,如果是积极请输出0,如果是消极请输出1(注意!!!:除了0和1以外,请不要输出任何其他文字或英文):
${question}
"""
argParser = ArgumentParser()
argParser.add_argument(
    "--model_path",
    type=str,
    default="/data/private/chenweize/checkpoints/Meta-Llama-3-8B-Instruct",
)
device = "cuda:4"
args = argParser.parse_args()
llama3 = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto").to(
    device
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, torch_dtype="auto")


dataset = []
with open("Dataset/test.txt") as f:
    for line in f:
        data = line.split()
        dataset.append({"label": data[0], "question": "".join(data[1:])})
sum = 0
right = 0
for data in dataset:
    sum += 1
    prompt = Template(prompt_template).safe_substitute({"question": data["question"]})
    messages_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt"
    ).to(device)

    response = llama3.generate(messages_input, max_new_tokens=1000, do_sample=True)
    response = tokenizer.batch_decode(response)
    response = response[0].split("<|end_header_id|>")[-1].strip().strip("<|eot_id|>")
    try:
        if int(response) == int(data["label"]):
            right += 1
    except:
        pass
    print(f"{response}:{data['label']}-{right}")

print(f"acc: {right/sum}")
