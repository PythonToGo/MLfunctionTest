## All codes are from ML and Task.

# Function to calculate non-embedding parameters
def calculate_non_embedding_params(dim_embd, n_heads, n_layers, dim_feedforward):
    mha_params_per_head = 3 * (dim_embd * dim_embd // n_heads)
    mha_params_per_layer = mha_params_per_head * n_heads
    output_proj_params = dim_embd * dim_embd
    ffn_params_per_layer = (dim_embd * dim_feedforward) + (dim_feedforward * dim_embd)
    total_params_per_layer = mha_params_per_layer + output_proj_params + ffn_params_per_layer
    return total_params_per_layer * n_layers

# Function to calculate FLOPs for matrix multiplication
def calculate_flops(i, j, k):
    return 2 * i * j * k

# Function to calculate FLOPs per token
def calculate_flops_per_token(dim_embd, n_heads, n_layers, dim_feedforward):
    qkv_flops = 3 * calculate_flops(dim_embd, dim_embd, dim_embd // n_heads)
    output_proj_flops = calculate_flops(dim_embd, dim_embd, dim_embd)
    mha_flops_per_layer = (qkv_flops + output_proj_flops) * n_heads
    attention_score_flops_per_head = calculate_flops(dim_embd // n_heads, dim_embd // n_heads, dim_embd // n_heads)
    attention_flops_per_layer = attention_score_flops_per_head * n_heads
    ffn_flops_1 = calculate_flops(dim_embd, dim_embd, dim_feedforward)
    ffn_flops_2 = calculate_flops(dim_embd, dim_feedforward, dim_embd)
    ffn_flops_per_layer = ffn_flops_1 + ffn_flops_2
    total_flops_per_layer = mha_flops_per_layer + attention_flops_per_layer + ffn_flops_per_layer
    return total_flops_per_layer * n_layers

# Function to estimate training cost per token
def estimate_training_cost_per_token(total_non_embedding_params):
    return 6 * total_non_embedding_params
