import tiktoken
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gpt2.config import Config
from gpt2.model import GPT2


class TinyShakespeareDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Trainer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def create_dataset(self):
        """
        1. Load the text file
        2. tokenize the data
        3. Create batches
        4. Add EOS token on both input and output and add eos in tokenizer too
        """
        text = open(self.config.data_path, "r").read()
        print("Tokenizing the data...")
        tokens = self.tokenizer.encode(text)

        # create input and output
        inputs = []
        targets = []
        eos_token_id = self.tokenizer.eot_token

        for i in range(0, len(tokens), self.config.seq_length - 1):
            # seq length -1 since we will also add <eos> token
            inp = tokens[i : i + self.config.seq_length - 1]
            out = tokens[i + 1 : i + self.config.seq_length]

            # add eot id
            inp.append(eos_token_id)
            out.append(eos_token_id)
            if len(inp) != self.config.seq_length or len(out) != self.config.seq_length:
                # if the last batch doesn't match seq length, we just ignore. But we can pad and use that
                continue
            inputs.append(torch.tensor(inp, dtype=torch.long))
            targets.append(torch.tensor(out, dtype=torch.long))

        # zip input andoutput
        dataset = list(zip(inputs, targets))

        return dataset

    def create_dataloader(self):
        data = self.create_dataset()
        dataset = TinyShakespeareDataset(data)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        return dataloader

    def create_model(self):
        return GPT2.from_pretrained("gpt2")

    def train(self):
        # create model
        model = self.create_model()
        model = model.to(self.config.device)

        # define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.eps,
        )

        # create dataloader
        dataloader = self.create_dataloader()

        for epoch in range(self.config.num_epochs):
            # Use tqdm to create a progress bar
            with tqdm(
                enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{self.config.num_epochs}"
            ) as progress_bar:
                for i, (inputs, outputs) in progress_bar:
                    # get outputs
                    inputs, outputs = inputs.to(self.config.device), outputs.to(self.config.device)

                    logits = model(inputs)
                    logits = logits.view(-1, logits.size(-1))
                    outputs = outputs.view(-1)

                    loss = criterion(logits, outputs)
                    # backward prop
                    optimizer.zero_grad()
                    loss.backward()
                    # gradient clipping to avoid exploding gradients
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    # update progress bar with the current loss
                    progress_bar.set_postfix(loss=loss.item())

                    # save model and config at the end of each epoch
                    if i == len(dataloader) - 1:
                        torch.save(model.state_dict(), self.config.model_path)
                        # Assuming save_config is a method of Config class to save the config
                        # self.config.save_config(self.config.config_path)


if __name__ == "__main__":
    trainer = Trainer(Config())
    trainer.train()
