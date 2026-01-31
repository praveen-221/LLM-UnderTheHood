import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Data preprocessing
#####################################

class GPTDatasetV1(Dataset):
    def __init__(self, input_txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        # tokenize the input sentence
        token_ids = tokenizer.encode(input_txt, allowed_special={"<|endoftext|>"})

        # use sliding window to create the input and target chunks
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i+max_length]
            target_chunk = token_ids[i+1 : i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    
    def __len__(self):
        return len(self.input_ids)

def create_dataloader_V1(input_text, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # create dataset
    train_dataset = GPTDatasetV1(input_text, tokenizer, max_length, stride)

    # create dataloader
    dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )
    return dataloader

#####################################
# Multi-Head Attention
#####################################

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim_in, dim_out, total_tokens, dropout_rate, num_heads, qkv_bias=False):
        super().__init__()
        assert(dim_out % num_heads == 0), "dim_out must be divisible by num_heads"

        # add the variables to self state that will be used later
        self.d_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads    # spliting weight matrix using number of heads (ex: d_out = 4, heads = 2 => then each head will have ..x2 dimension) 

        self.W_queries = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_keys = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_values = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.projection = torch.nn.Linear(dim_out, dim_out)     # Linear layer to combine head outputs
        self.dropout_layer = torch.nn.Dropout(dropout_rate)
        # registr_buffer is used to register a buffer that should not be considered a model parameter which are arbitary values used for computation
        self.register_buffer('causal_mask', torch.triu(torch.ones(total_tokens, total_tokens), diagonal=1))
    
    def forward(self, x):
        num_batch, num_tokens, dim_in = x.shape
        
        # calculate the q,k,V values w.r.t each input token
        queries = self.W_queries(x)
        keys = self.W_keys(x)
        values = self.W_values(x)

        # implicitly split the matrix by adding a `num_heads` dimension
        # Unroll/reshape last dim: (num_batch, num_tokens, d_out) -> (num_batch, num_tokens, num_heads, head_dim)
        queries = queries.view(num_batch, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(num_batch, num_tokens, self.num_heads, self.head_dim)
        values = values.view(num_batch, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (num_batch, num_tokens, num_heads, head_dim) -> (num_batch, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Dot product for each head is computed by matrix multiplication of (num_tokens x head_dim) matrices
        attn_scores = queries @ keys.transpose(2, 3)

        # apply causal mask to the attention scores inplace without a variable memory space used
        attn_scores.masked_fill_(
            self.causal_mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # apply dropout to the attention weights calculated
        attn_weights = self.dropout_layer(attn_weights)

        # calculate context vectores for each input token
        context_vectors = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vectors = context_vectors.contiguous().view(num_batch, num_tokens, self.d_out)
        context_vectors = self.projection(context_vectors)   # optional projection

        return context_vectors

#####################################
# Transformer Architecture
#####################################

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-5):
        super().__init__()
        self.eps = eps              # small value to avoid division by zero error while normalizing with variance
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        variance = x.var(dim = -1, keepdim = True, unbiased = False)
        norm_output = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * norm_output + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
                (x + 0.044715 * torch.pow(x, 3))
            )
        )

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["embedding_dim"], 4*config["embedding_dim"]),
            GELU(),
            nn.Linear(4*config["embedding_dim"], config["embedding_dim"])
        )
    
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_layer = MultiHeadAttention(
            dim_in = config["embedding_dim"],
            dim_out = config["embedding_dim"],
            total_tokens = config["contextWindow_len"],
            dropout_rate = config["dropout_rate"],
            num_heads = config["num_heads"],
            qkv_bias = config["qkv_bias"]
        )
        self.layerNorm1 = LayerNorm(config["embedding_dim"], eps = 1e-5)
        self.layerNorm2 = LayerNorm(config["embedding_dim"], eps = 1e-5)
        self.feedForward_layer = FeedForward(config)
        self.dropout_layer = nn.Dropout(config["dropout_rate"])
    
    def forward(self, x):
        # Attention block
        shortcut = x        # Shortcut connection for attention block
        x = self.layerNorm1(x)
        x = self.attention_layer(x)     # Shape [batch_size, num_tokens, emb_size]
        x = self.dropout_layer(x)
        # Add the original input back
        x = x + shortcut

        # Feed Forward block
        shortcut = x        # Shortcut connection for feed forward block
        x = self.layerNorm2(x)
        x = self.feedForward_layer(x)
        x = self.dropout_layer(x)
        x = x + shortcut

        return x

#####################################
# GPT-2 Architecture
#####################################

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Embedding layers
        self.token_embedding_layer = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.position_embedding_layer = nn.Embedding(config["contextWindow_len"], config["embedding_dim"])
        self.dropout_layer = nn.Dropout(config["dropout_rate"])

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["num_transformerLayers"])]
        )

        # Layer Norm
        self.final_norm = LayerNorm(config["embedding_dim"])
        # layer to convert token ids into text
        self.output_head = nn.Linear(
            config["embedding_dim"], config["vocab_size"], bias=False
        )

    def forward(self, input_sentence):
        # extracting the number of sentences present in the batch(batch_size) & number of tokens in each sentence(sequence_len)
        batch_size, sequence_len = input_sentence.shape
        token_embeddings = self.token_embedding_layer(input_sentence)
        position_embeddings = self.position_embedding_layer(torch.arange(sequence_len, device=input_sentence.device))
        x = token_embeddings + position_embeddings
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits

##################################################################
# utility function to generate token ids for given input sentence
##################################################################

def generate_text_tokens(model, input_ids, max_new_tokens, context_window):
    # input_ids is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10 then only the last 5 tokens are used as context
        ids_context_size = input_ids[:, -context_window:]

        with torch.no_grad():
            logits = model(ids_context_size)
        
        # Focus only on the last token
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)   # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        next_id = torch.argmax(probabilities, dim=-1, keepdim=True)     # (batch, 1)

        # Append sampled index to the running sequence
        input_ids = torch.cat((input_ids, next_id), dim=-1)   # (batch, n_tokens+1)
        
    return input_ids