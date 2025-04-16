# GPT-2 Implementation

This repository contains a minimal implementation of GPT-2 for training and inference, using PyTorch.

## Setup

1. **Clone the repository**  
   ```bash
   git clone git@github.com:TimilsinaBimal/gpt2.git
   cd gpt2
   ```

2. **Install dependencies**  
   Make sure you have Python and pip installed along with `torch`, `transformers` and `tiktoken` installed.

3. **Prepare your data**  
   Place your training text file (e.g., `tiny_shakespeare.txt`) in the desired location. Update the `data_path` in your config if needed.

---

## Training

To train the model:

```bash
python -m gpt2.train
```

- The training script will:
  - Load and tokenize your dataset.
  - Train the GPT-2 model using the specified configuration.
  - Save the model weights at the end of each epoch to the path specified in your config.

**Configuration:**  
Edit the [`gpt2/config.py`](gpt2/config.py) file to adjust parameters such as `data_path`, `batch_size`, `seq_length`, `learning_rate`, `num_epochs`, and `model_path`.

---

## Inference

To run inference with a pretrained model:

1. Ensure you have a trained model checkpoint at the path specified in your config.
2. Use the inference script:

```bash
python -m gpt2.inference
```

- The script will:
  - Load the pretrained GPT-2 model.
  - Generate text based on your input prompt (modify the script to set your prompt).

---

## Notes

- The tokenizer uses `tiktoken` with the GPT-2 vocabulary.
- Model checkpoints and configuration files are saved as specified in your config.
- For custom datasets or different model sizes, update the config and scripts accordingly.

---

## File Structure

- [`gpt2/train.py`](gpt2/train.py): Training pipeline.
- [`gpt2/inference.py`](gpt2/inference.py): Inference/generation script.
- [`gpt2/model.py`](gpt2/model.py): Model definition and loading.
- [`gpt2/config.py`](gpt2/config.py): Configuration class.

---

## Troubleshooting

- Ensure your data file path is correct in the config.
- If CUDA is available, set `device` in config to `"cuda"` for faster training.

---

## License

Apache 2.0 License

*Disclaimer: This README file is generated using AI*
