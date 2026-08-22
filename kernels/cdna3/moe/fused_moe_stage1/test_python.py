import torch

D_HIDDEN = 5
D_EXPERT = 4
SWIZZLE_GRANULARITY = 2

gate_weights = torch.randn((D_EXPERT, D_HIDDEN))
up_weights = torch.randn((D_EXPERT, D_HIDDEN))

def interleave_gate_up(gate, up, granularity):
    d_expert, d_hidden = gate.shape
    assert d_expert % granularity == 0

    # Split the D_EXPERT dim into (num_chunks, granularity)
    gate_chunks = gate.reshape(d_expert // granularity, granularity, d_hidden)
    up_chunks   = up.reshape(d_expert // granularity, granularity, d_hidden)

    # Stack so each chunk pair is adjacent: (num_chunks, 2, granularity, d_hidden)
    # dim=1 ordering: [up_chunk, gate_chunk] to match "[up_0:64, gate_64:128, ...]"
    interleaved = torch.stack([up_chunks, gate_chunks], dim=1)

    # Flatten back to (2 * D_EXPERT, D_HIDDEN)
    return interleaved.reshape(2 * d_expert, d_hidden)

swizzled = interleave_gate_up(gate_weights, up_weights, SWIZZLE_GRANULARITY)

print(gate_weights)
print(up_weights)
print(swizzled)
