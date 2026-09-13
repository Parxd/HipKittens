import torch
import aiter
from aiter.fused_moe_bf16_asm import moe_sorting_ck
import tk_kernel

inter_dim = 64
model_dim = 256
topk = 2

num_tokens = 4
num_experts = 8
block_m = 32
block_n = 128
WEIGHT_SWIZZLE_GRANULARITY = block_n / 2
fp8 = torch.float8_e4m3fnuz


def interleave_gate_up(w_gate: torch.Tensor, w_up: torch.Tensor, block_size: int) -> torch.Tensor:
    assert w_gate.shape == w_up.shape, "gate and up must have the same shape"
    num_experts, inter_dim, model_dim = w_gate.shape
    assert inter_dim % block_size == 0, "inter_dim must be divisible by block_size"

    num_blocks = inter_dim // block_size
    # Reshape into blocks: (num_experts, num_blocks, block_size, model_dim)
    gate_blocks = w_gate.view(num_experts, num_blocks, block_size, model_dim)
    up_blocks = w_up.view(num_experts, num_blocks, block_size, model_dim)
    # Stack gate/up along a new axis right after num_blocks:
    # (num_experts, num_blocks, 2, block_size, model_dim)
    interleaved = torch.stack((gate_blocks, up_blocks), dim=2)
    # Flatten (num_blocks, 2, block_size) -> (2 * inter_dim)
    interleaved = interleaved.reshape(num_experts, 2 * inter_dim, model_dim)
    return interleaved


def moe_stage1_reference(
    activations_fp8: torch.Tensor,
    activation_scales: torch.Tensor,
    interleaved_weights_fp8: torch.Tensor,
    weight_scales: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    *,
    topk: int,
    block_m: int,
    swizzle_granularity: int,
) -> torch.Tensor:
    """Reference for a8w8 per-token/per-channel MoE stage 1 with SwiGLU."""
    num_tokens, model_dim = activations_fp8.shape
    num_experts, packed_inter_dim, weight_model_dim = interleaved_weights_fp8.shape
    assert model_dim == weight_model_dim
    assert packed_inter_dim % 2 == 0
    assert packed_inter_dim % (2 * swizzle_granularity) == 0

    inter_dim = packed_inter_dim // 2
    assert weight_scales.shape == (num_experts, packed_inter_dim)
    assert activation_scales.numel() == num_tokens

    num_swizzle_blocks = inter_dim // swizzle_granularity
    weights = interleaved_weights_fp8.float().reshape(
        num_experts, num_swizzle_blocks, 2, swizzle_granularity, model_dim
    )
    gate_weights = weights[:, :, 0].reshape(num_experts, inter_dim, model_dim)
    up_weights = weights[:, :, 1].reshape(num_experts, inter_dim, model_dim)

    activations = activations_fp8.float()
    activation_scales = activation_scales.float().reshape(num_tokens)
    weight_scales = weight_scales.float()
    output = torch.zeros(
        (num_tokens * topk, inter_dim), dtype=torch.float32, device=activations.device
    )

    num_valid_tiles = int(num_valid_ids.item()) // block_m
    for tile_index in range(num_valid_tiles):
        packed_ids = sorted_token_ids[tile_index * block_m : (tile_index + 1) * block_m]
        token_ids = (packed_ids & 0x00FFFFFF).long()
        topk_slots = ((packed_ids >> 24) & 0xFF).long()
        valid = token_ids < num_tokens
        if not valid.any():
            continue

        expert = int(sorted_expert_ids[tile_index].item())
        token_ids = token_ids[valid]
        dequantized_activations = activations[token_ids] * activation_scales[token_ids, None]
        gate = dequantized_activations @ gate_weights[expert].T
        up = dequantized_activations @ up_weights[expert].T
        gate *= weight_scales[expert, :inter_dim]
        up *= weight_scales[expert, inter_dim:]

        output[token_ids * topk + topk_slots[valid]] = torch.nn.functional.silu(gate) * up

    return output.to(torch.bfloat16)


# sanity checks
debug = True

if debug:
    torch.set_printoptions(profile="full", sci_mode=False)

    hidden_states = torch.ones(num_tokens, model_dim, dtype=torch.bfloat16, device="cuda")
    w1_gate = torch.ones(num_experts, inter_dim, model_dim, dtype=torch.bfloat16, device="cuda")
    w1_up = torch.ones(num_experts, inter_dim, model_dim, dtype=torch.bfloat16, device="cuda")
    w2 = torch.ones(num_experts, model_dim, inter_dim, dtype=torch.bfloat16, device="cuda")
    router_logits = torch.ones(num_tokens, num_experts, device="cuda")
else:
    hidden_states = torch.randn(num_tokens, model_dim, dtype=torch.bfloat16, device="cuda")
    w1_gate = torch.randn(num_experts, inter_dim, model_dim, dtype=torch.bfloat16, device="cuda")
    w1_up = torch.randn(num_experts, inter_dim, model_dim, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(num_experts, model_dim, inter_dim, dtype=torch.bfloat16, device="cuda")
    router_logits = torch.randn(num_tokens, num_experts, device="cuda")

hidden_states_fp8 = hidden_states.to(fp8)
w1_gate_fp8 = w1_gate.to(fp8)
w1_up_fp8 = w1_up.to(fp8)
w1_fp8 = torch.concat((w1_gate_fp8, w1_up_fp8), dim=1)
w2_fp8 = w2.to(fp8)

topk_weights, topk_ids = torch.topk(router_logits.softmax(dim=-1), k=topk, dim=-1)
topk_ids = topk_ids.to(torch.int32)

sorted_ids, _sorted_weights, sorted_expert_ids, num_valid_ids, _moe_buf = (
    moe_sorting_ck(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        torch.bfloat16,
        block_size=block_m,
        expert_mask=None,
    )
)
torch.cuda.synchronize()

out_ref = torch.empty((num_tokens * topk, inter_dim), dtype=torch.bfloat16, device="cuda")
out_test = torch.empty((num_tokens * topk, inter_dim), dtype=torch.bfloat16, device="cuda")

if debug:
    a1_scale = torch.ones(num_tokens, 1, dtype=torch.float32, device="cuda")
    w1_scale = torch.ones(num_experts, 1, inter_dim * 2, dtype=torch.float32, device="cuda")
else:
    a1_scale = torch.rand(num_tokens, 1, dtype=torch.float32, device="cuda")
    w1_scale = torch.rand(num_experts, 1, inter_dim * 2, dtype=torch.float32, device="cuda")

if 0:
    aiter.ck_moe_stage1_fwd(
        hidden_states=hidden_states_fp8,
        w1=w1_fp8,
        w2=w2_fp8,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out_ref,
        topk=topk,
        kernelName="",
        w1_scale=w1_scale,
        a1_scale=a1_scale,
        block_m=32,
        sorted_weights=_sorted_weights,
        quant_type=aiter.QuantType.per_Token,
        activation=aiter.ActivationType.Swiglu
    )
    torch.cuda.synchronize()

interleaved = interleave_gate_up(w1_gate_fp8, w1_up_fp8, WEIGHT_SWIZZLE_GRANULARITY)
out_ref = moe_stage1_reference(
    hidden_states_fp8,
    a1_scale,
    interleaved,
    w1_scale,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    topk=topk,
    block_m=block_m,
    swizzle_granularity=WEIGHT_SWIZZLE_GRANULARITY,
)
tk_kernel.call(
    hidden_states_fp8,
    a1_scale.reshape((num_tokens)),
    interleaved,
    w1_scale.reshape((num_experts, inter_dim * 2)),
    out_test,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids[0] / block_m 
)
torch.cuda.synchronize()

torch.testing.assert_close(out_test, out_ref, rtol=2e-2, atol=2e-2)

if 1:
    print(topk_ids)
    print(sorted_ids[:] & 0xFFFFFF)
    print(out_ref)
    print(out_test)