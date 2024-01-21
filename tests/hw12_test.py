import hw12

def test_calculate_non_embedding_params():
    # Test with example parameters
    dim_embd = 768
    n_heads = 12
    n_layers = 12
    dim_feedforward = 3072
    result = hw12.calculate_non_embedding_params(dim_embd, n_heads, n_layers, dim_feedforward)
    assert result == 84934656  # Expected result

def test_calculate_flops_per_token():
    # Test with example parameters
    dim_embd = 768
    n_heads = 12
    n_layers = 12
    dim_feedforward = 3072
    result = hw12.calculate_flops_per_token(dim_embd, n_heads, n_layers, dim_feedforward)
    assert result == 250123124736  # Expected result

def test_estimate_training_cost_per_token():
    # Test with example parameters
    total_non_embedding_params = 84934656
    result = hw12.estimate_training_cost_per_token(total_non_embedding_params)
    assert result == 509607936  # Expected result
