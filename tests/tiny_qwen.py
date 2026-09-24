import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

def make_tiny(seed=0, attn="eager"):
    torch.manual_seed(seed)
    cfg = Qwen2_5_VLConfig(
        vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        image_token_id=300, video_token_id=301, vision_start_token_id=302, vision_end_token_id=303,
        bos_token_id=1, eos_token_id=2,
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=64, patch_size=14, spatial_merge_size=2,
                           temporal_patch_size=2, window_size=56, fullatt_block_indexes=[0], in_chans=3),
    )
    cfg._attn_implementation = attn
    m = Qwen2_5_VLForConditionalGeneration(cfg).eval()
    return m

def make_inputs(n_text_before=5, n_text_after=6, grid=(1, 4, 4)):
    t, h, w = grid
    n_img = t * h * w // 4
    ids = [5 + i for i in range(n_text_before)] + [302] + [300] * n_img + [303] + [40 + i for i in range(n_text_after)]
    input_ids = torch.tensor([ids])
    pv = torch.randn(t * h * w, 3 * 2 * 14 * 14)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids),
            "pixel_values": pv, "image_grid_thw": torch.tensor([list(grid)])}

if __name__ == "__main__":
    m = make_tiny(); inp = make_inputs()
    out = m(**inp, output_hidden_states=True, output_attentions=True, use_cache=False, return_dict=True)
    print(out.logits.shape, len(out.hidden_states), out.attentions[0].shape)
    print([n for n, _ in m.named_children()], type(m.model).__name__, hasattr(m.model, "norm"), len(m.model.layers))
