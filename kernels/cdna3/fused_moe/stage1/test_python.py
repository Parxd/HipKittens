import torch
import aiter
from aiter.fused_moe_bf16_asm import moe_sorting_ck

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
    # dim=1 ordering: [gate_chunk, up_chunk] to match "[gate_0:64, up_64:128, ...]"
    interleaved = torch.stack([gate_chunks, up_chunks], dim=1)

    # Flatten back to (2 * D_EXPERT, D_HIDDEN)
    return interleaved.reshape(2 * d_expert, d_hidden)

swizzled = interleave_gate_up(gate_weights, up_weights, SWIZZLE_GRANULARITY)

print(gate_weights)
print(up_weights)
print(swizzled)



torch.set_default_device("cuda")
fp8_dtype = torch.float8_e4m3fnuz

num_tokens = 4
model_dim = 128
inter_dim = 256
num_experts = 8
topk = 2

hidden_states = torch.randn(num_tokens, model_dim, dtype=torch.bfloat16, device="cuda")
hidden_states_fp8 = hidden_states.to(fp8_dtype)

w1 = torch.randn(num_experts, inter_dim * 2, model_dim, dtype=torch.bfloat16, device="cuda")
w1_fp8 = w1.to(fp8_dtype)
w2 = torch.randn(num_experts, model_dim, inter_dim, dtype=torch.bfloat16, device="cuda")
w2_fp8 = w2.to(fp8_dtype)

router_logits = torch.randn(num_tokens, num_experts, device="cuda")
topk_weights, topk_ids = torch.topk(router_logits.softmax(dim=-1), k=topk, dim=-1)
topk_ids = topk_ids.to(torch.int32)

sorted_ids, _sorted_weights, sorted_expert_ids, num_valid_ids, _moe_buf = (
    moe_sorting_ck(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        torch.bfloat16,
        block_size=32,
        expert_mask=None,
    )
)

out = torch.empty((sorted_ids.numel(), inter_dim), dtype=torch.bfloat16, device="cuda")
a1_scale = torch.rand(num_tokens, 1, dtype=torch.float32, device="cuda")
w1_scale = torch.rand(num_experts, 1, inter_dim * 2, dtype=torch.float32, device="cuda")

aiter.ck_moe_stage1_fwd(
    hidden_states=hidden_states_fp8,
    w1=w1_fp8,
    w2=w2_fp8,
    sorted_token_ids=sorted_ids,
    sorted_expert_ids=sorted_expert_ids,
    num_valid_ids=num_valid_ids,
    out=out,
    topk=topk,
    kernelName="",
    w1_scale=w1_scale,
    a1_scale=a1_scale,
    block_m=32,
    sorted_weights=_sorted_weights,
    quant_type=aiter.QuantType.per_Token,
    activation=aiter.ActivationType.Swiglu
)

torch.set_printoptions(profile="full")

print(topk_ids)
print(sorted_ids)
print(out)