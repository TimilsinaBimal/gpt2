# get some text, tokenize and generate output
import tiktoken
import torch
import torch.nn.functional as F

from gpt2.model import GPT2

torch.manual_seed(42)
tokenizer = tiktoken.get_encoding("gpt2")
model = GPT2.from_pretrained("gpt2")


def generate_text(model, tokenizer, prompt, max_length=30):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    print(prompt, end="", flush=True)
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
            # take last output token
            logits = logits[:, -1, :]  # (batch_size, vocab_size)
            probs = F.softmax(logits, dim=-1)

            # topk probabilities: get top 5 probabilities and their indices
            # takes top k highest probabilities tokens
            # resamples them so that their sum becomes 1
            # and then samples from them
            # dim=-1 coz, we sample of vocab size dimension
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # batch_size, 50

            # sampling from topk_probs. This will normally follow normal distribition on sampling over multiple samples
            token_indices = torch.multinomial(topk_probs, num_samples=1)  # batch_size, 1

            # multinomial returns indices and not actual tokens. So we need to get the actual tokens from topk_indices
            sampled_token = torch.gather(topk_indices, -1, token_indices)  # batch_size, 1

            # append the sampled token to input
            input_ids = torch.cat([input_ids, sampled_token], dim=1)
            if sampled_token == tokenizer._special_tokens.get("<|endoftext|>"):
                break
            # now decode tokens
            for idx in range(input_ids.size()[0]):  # batch size
                generated_text = tokenizer.decode(sampled_token[idx].tolist())

                print(generated_text, end="", flush=True)


generate_text(model, tokenizer, "Hello, I am a language model,", max_length=100)
