import tiktoken
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

##############################################
# Loading pre-trained weights for GPT2
##############################################

# utility function which converts tensorflow weights into pytorch trainable parameters
def assign_tf_torch(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt_instance(gptInstance, params):
    # assign embedding layer parameter weights
    gptInstance.token_embedding_layer.weight = assign_tf_torch(
        gptInstance.token_embedding_layer.weight, 
        params["wte"]
    )
    gptInstance.position_embedding_layer.weight = assign_tf_torch(
        gptInstance.position_embedding_layer.weight, 
        params["wpe"]
    )
    
    # assign transformer block parameter weights
    for b in range(len(params["blocks"])):
        # MHA layer paramters
        # GPT2 uses a combined qkv weight matrix which needs to be split inorder to load weights
        q_W, k_W, v_W = np.split(
            params["blocks"][b]["attn"]["c_attn"]["w"], indices_or_sections=3, axis=-1
        )
        # W_queries weight params
        gptInstance.transformer_blocks[b].attention_layer.W_queries.weight = assign_tf_torch(
            gptInstance.transformer_blocks[b].attention_layer.W_queries.weight, 
            q_W.T
        )
        # W_keys weight params
        gptInstance.transformer_blocks[b].attention_layer.W_keys.weight = assign_tf_torch(
            gptInstance.transformer_blocks[b].attention_layer.W_keys.weight, 
            k_W.T
        )
        # W_values weight params
        gptInstance.transformer_blocks[b].attention_layer.W_values.weight = assign_tf_torch(
            gptInstance.transformer_blocks[b].attention_layer.W_values.weight, 
            v_W.T
        )

        # bias
        # GPT2 uses a combined qkv bias matrix which needs to be split inorder to load bias
        q_b, k_b, v_b = np.split(
            params["blocks"][b]["attn"]["c_attn"]["b"], indices_or_sections=3, axis=-1
        )
        # W_queries bias params
        gptInstance.transformer_blocks[b].attention_layer.W_queries.bias = assign_tf_torch(
            gptInstance.transformer_blocks[b].attention_layer.W_queries.bias,
            q_b
        )
        # W_keys bias params
        gptInstance.transformer_blocks[b].attention_layer.W_keys.bias = assign_tf_torch(
            gptInstance.transformer_blocks[b].attention_layer.W_keys.bias,
            k_b
        )
        # W_values bias params
        gptInstance.transformer_blocks[b].attention_layer.W_values.bias = assign_tf_torch(
            gptInstance.transformer_blocks[b].attention_layer.W_values.bias,
            v_b
        )

        # projection layer weight and bias
        gptInstance.transformer_blocks[b].attention_layer.projection.weight = assign_tf_torch(
            gptInstance.transformer_blocks[b].attention_layer.projection.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gptInstance.transformer_blocks[b].attention_layer.projection.bias = assign_tf_torch(
            gptInstance.transformer_blocks[b].attention_layer.projection.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        # LayerNorm1 scale(g) and shift(b) parameters
        gptInstance.transformer_blocks[b].layerNorm1.scale = assign_tf_torch(
           gptInstance.transformer_blocks[b].layerNorm1.scale,
           params["blocks"][b]["ln_1"]["g"]
        )
        gptInstance.transformer_blocks[b].layerNorm1.shift = assign_tf_torch(
           gptInstance.transformer_blocks[b].layerNorm1.shift,
           params["blocks"][b]["ln_1"]["b"]
        )

        # LayerNorm2 scale(g) and shift(b) parameters
        gptInstance.transformer_blocks[b].layerNorm2.scale = assign_tf_torch(
           gptInstance.transformer_blocks[b].layerNorm2.scale,
           params["blocks"][b]["ln_2"]["g"]
        )
        gptInstance.transformer_blocks[b].layerNorm2.shift = assign_tf_torch(
           gptInstance.transformer_blocks[b].layerNorm2.shift,
           params["blocks"][b]["ln_2"]["b"]
        )

        # Feed forward layer weight and bias paramters
        # Linear layer 1
        gptInstance.transformer_blocks[b].feedForward_layer.layers[0].weight = assign_tf_torch(
            gptInstance.transformer_blocks[b].feedForward_layer.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gptInstance.transformer_blocks[b].feedForward_layer.layers[0].bias = assign_tf_torch(
            gptInstance.transformer_blocks[b].feedForward_layer.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        # Linear layer 2
        gptInstance.transformer_blocks[b].feedForward_layer.layers[2].weight = assign_tf_torch(
            gptInstance.transformer_blocks[b].feedForward_layer.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gptInstance.transformer_blocks[b].feedForward_layer.layers[2].bias = assign_tf_torch(
            gptInstance.transformer_blocks[b].feedForward_layer.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )
    
    # assign parameters to final layer norm
    gptInstance.final_norm.scale = assign_tf_torch(
        gptInstance.final_norm.scale,
        params["g"]
    )
    gptInstance.final_norm.shift = assign_tf_torch(
       gptInstance.final_norm.shift,
       params["b"]
    )

    # assign parameters to final output_head layer
    gptInstance.output_head.weight = assign_tf_torch(
        gptInstance.output_head.weight,
        params["wte"]
    )

def text_to_token_ids(input_sentence, tokenizer):
    encoded_text = tokenizer.encode(input_sentence, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded_text).unsqueeze(0)      # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    decoded_list = token_ids.squeeze(0).tolist()    # remove batch dimension
    return tokenizer.decode(decoded_list)

################################################################################################
# utility function to generate token ids for given input sentence with sampling techniques
################################################################################################

def generate_tokens_with_sampling(model, input_ids, max_new_tokens, context_window, top_k = None, temperature = 0.0, eos_id = None):
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

        # Filter top k values from the logits
        if top_k is not None:
            topK_logits, topK_indices = torch.topk(logits, k=top_k)
            minimum_value = topK_logits[:, -1]
            logits = torch.where(
                condition = logits < minimum_value,
                input = torch.tensor(-torch.inf),
                other = logits
            )
        
        # apply temperature scaling
        if temperature > 0.0:
            scaled_logits = logits / temperature
            probabilities = torch.softmax(scaled_logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)

        else:
            # Get the idx of the vocab entry with the highest probability value
            next_id = torch.argmax(logits, dim=-1, keepdim=True)     # (batch, 1)
        
        # if the next token generated is qual to end of sequence defined, then stop the text generation process
        if next_id == eos_id:
            break

        # Append sampled index to the running sequence
        input_ids = torch.cat((input_ids, next_id), dim=-1)   # (batch, n_tokens+1)
        
    return input_ids

################################################################################################
# utility functions to train LLM to generate text
################################################################################################

# utiltiy functions to calculate loss for each batch 
def calc_loss_batch(input_data, target_data, model, device):
    # loading data to the same device as model
    input_data, target_data = input_data.to(device), target_data.to(device)
    logits = model(input_data)
    cross_entropy_loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_data.flatten())
    return cross_entropy_loss

# utility function to calculate loss for entire dataset
def calc_loss_dataloader(input_dataloader, model, device, num_batches = None):
    total_loss = 0.
    if len(input_dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(input_dataloader)
    else: 
        # if num_batches passed exceeds the number of batches in the data loader or less than the number in data loader 
        # change the number of batches to match the total number of batches in the data loader or num_batches passed whichever is minimum
        num_batches = min(num_batches, len(input_dataloader))
    for i, (x, y) in enumerate(input_dataloader):
        if i < num_batches:
            batch_loss = calc_loss_batch(x, y, model, device)
            total_loss += batch_loss.item()
        else:
            break

    return total_loss / num_batches     # returning average loss over the dataset

# utility function to evaluate model
def evaluate_training(model, train_dataloader, validation_dataloader, device, num_eval_batches):
    model.eval()
    with torch.no_grad():
        eval_train_loss = calc_loss_dataloader(train_dataloader, model, device, num_eval_batches)
        eval_validation_loss = calc_loss_dataloader(validation_dataloader, model, device, num_eval_batches)
    return eval_train_loss, eval_validation_loss

# utility function to generate text tokens
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

# utility function to test text generation
def generate_and_print_text(model, sample_input, tokenizer, device, max_tokens=50):
    model.eval()
    context_window_len = model.position_embedding_layer.weight.shape[0]
    sample_tokens = text_to_token_ids(sample_input, tokenizer).to(device)
    # number of tokens to be generated can be passed as a varaible 
    with torch.no_grad():
        generated_tokens_sample = generate_text_tokens(model, sample_tokens, max_new_tokens= max_tokens, context_window= context_window_len)
    generated_text_sample = token_ids_to_text(generated_tokens_sample, tokenizer)
    # print(f"="*150)
    print(f"\nModel output: {generated_text_sample.replace("\n", " ")}")    # replaces new line is generated by model to space for readability 
    print(f"="*150)

def model_training_simple(
        model,
        train_dataloader,
        validation_dataloader,
        optimizer,
        device,
        num_epochs,
        eval_frequency,     # defining the frequency of model evaluation(if n, evaluate model after every n iteration)
        eval_batch_iteration,     # controls the number of batches used for evaluation 
        sample_input,
        tokenizer,
        max_tokens
    ):
    train_loss, validation_loss, track_tokens_seen = [], [], []
    num_tokens_seen, global_step = 0, -1

    # iterating over the entire dataset for `num_epochs` time
    for epoch in range(num_epochs):
        model.train()

        # iterating over the training data with each batch as one step in training
        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()        # Reset gradients to zero 
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()             # Calculate loss gradients
            optimizer.step()            # update weights based on loss gradients
            num_tokens_seen += input_batch.numel()
            global_step += 1

            # evaluate model if number of steps reaches the frequency of evaluation(ex: for every 5th step)
            if global_step % eval_frequency == 0:
                step_training_loss, step_validation_loss = evaluate_training(
                    model, train_dataloader, validation_dataloader, device, eval_batch_iteration
                )
                train_loss.append(step_training_loss)
                validation_loss.append(step_validation_loss)
                track_tokens_seen.append(num_tokens_seen)
                print(f"Epoch {epoch + 1} [Step {global_step:03d}]:"
                      f"\tTraining Loss: {step_training_loss:.3f} \t Validation Loss: {step_validation_loss:.3f}")

        generate_and_print_text(model, sample_input, tokenizer, device, max_tokens)

    return train_loss, validation_loss, track_tokens_seen

#####################################
# function to plot graphs
#####################################

# function to plot the loss graph and save it as PDF
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.savefig("training_loss-plot.pdf")
    plt.show()
