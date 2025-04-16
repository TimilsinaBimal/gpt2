import math
import torch
import torch.nn.functional as F
from torch import nn

from gpt2.config import Config


class MLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        # no need to use approximate these days but since it was used in gpt2 we are using it here
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.n_embed % config.n_head == 0, "The value of n_head must be divisible by n_embed"

        self.d_k = self.d_q = self.d_v = config.n_embed // config.n_head

        # q, k, v all use same linear unit, concatenate
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.n_embed = config.n_embed
        self.n_head = config.n_head

        look_ahead_mask = torch.tril(
            torch.ones((config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )

        self.register_buffer("bias", look_ahead_mask)

    def attention(self, q, k, v, seq_length):
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / math.sqrt(k.size(-1))

        # mask the attention to restrict model from seeing future tokens
        attn = attn.masked_fill(self.bias[:, :, :seq_length, :seq_length] == 0, float("-inf"))
        # Why until seq length for last two dimensions? Because when inference, we might not get fixed sequence length

        attn = F.softmax(attn, dim=-1)

        y = torch.matmul(attn, v)
        return y

    def forward(self, x):
        batch_size, seq_length, embed_dimension = x.size()
        qkv = self.c_attn(x)  # multi-head calculation

        # now we need to calculate attention and mask the tokens
        q, k, v = qkv.split(split_size=self.n_embed, dim=2)

        q = q.view(batch_size, seq_length, self.n_head, embed_dimension // self.n_head).transpose(1, 2)
        # bring head in place of seq_length for parallel multiplicaion
        k = k.view(batch_size, seq_length, self.n_head, embed_dimension // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_head, embed_dimension // self.n_head).transpose(1, 2)

        # now calculate attention
        attn_out = self.attention(q, k, v, seq_length)

        # move head back to second last dim, since we need to convert this to original size now
        # current (batch_size, n_head, seq_length, embedding_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dimension)

        # final proj layer
        y = self.c_proj(attn_out)
        return y


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embed, eps=config.eps)
        self.attn = CausalAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed, eps=config.eps)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_layer = config.n_layer

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                wpe=nn.Embedding(config.seq_length, config.n_embed),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embed, eps=config.eps),
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_type: str):
        from transformers import GPT2LMHeadModel

        gpt_model = GPT2LMHeadModel.from_pretrained(model_type)
        # get state dict/weights
        hf_state_dict = gpt_model.state_dict()

        # load hf config and use params from there
        hf_config = gpt_model.config

        config = Config(
            block_size=hf_config.n_positions,
            vocab_size=hf_config.vocab_size,
            n_layer=hf_config.n_layer,
            n_head=hf_config.n_head,
            n_embed=hf_config.n_embd,
            seq_length=hf_config.n_positions,
            eps=hf_config.layer_norm_epsilon,
        )

        # lets also load the current model
        model = GPT2(config)
        state_dict = model.state_dict()

        # sd_keys = state_dict.keys()

        # remove masks and attention biases
        hf_sd_keys = hf_state_dict.keys()

        skip_suffixes = ".attn.masked_bias"
        hf_sd_keys = [k for k in hf_state_dict if not k.endswith(skip_suffixes)]  # buffers, not functinal masks

        # some parameters in hf models are transposed (IDK WHY). So lets fix them
        transposed_keys = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        # Now copy
        for key in hf_sd_keys:
            if any([key.endswith(k) for k in transposed_keys]):
                # make sure no issues with shape of both state dicts, make sure hf state dict is transposed
                assert state_dict[key].size() == hf_state_dict[key].size()[::-1]
                # transposed keys
                with torch.no_grad():
                    state_dict[key].copy_(hf_state_dict[key].T)
            else:
                assert state_dict[key].size() == hf_state_dict[key].size()
                with torch.no_grad():
                    state_dict[key].copy_(hf_state_dict[key])

        return model

    def forward(self, x):
        _, seq_length = x.size()  # batch_size, seq_length
        assert seq_length <= self.config.block_size, f"Sequence length cannot be greater than {seq_length}"

        # generate random numbers for positional encodings
        pos = torch.arange(0, seq_length, 1, dtype=x.dtype, device=x.device).unsqueeze(0)  # (1, seq_length)
        token_embed = self.transformer.wte(x)
        pos_embed = self.transformer.wpe(pos)
        x = token_embed + pos_embed
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # final classifier layer
        logits = self.lm_head(x)
        return logits
