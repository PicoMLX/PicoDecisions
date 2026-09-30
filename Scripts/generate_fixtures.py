"""Generate deterministic tiny-model fixtures with the unmodified Python MLX port.

Usage: python Scripts/generate_fixtures.py /path/to/laya-mlx
Reference revision: fc1df62828a3fedf4d8229fdac1cbd85f1cdf337
Requires mlx, numpy, tokenizers, safetensors, huggingface_hub.
"""
import json
from pathlib import Path
import sys

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from laya_mlx.model import DecisionModel, EncoderConfig
from laya_mlx.agent import Agent, collate_items

root = Path(__file__).resolve().parents[1] / "Tests/PicoDecisionsMLXTests/Fixtures/tiny"
root.mkdir(parents=True, exist_ok=True)
(root / "encoder").mkdir(exist_ok=True)
(root / "tokenizer").mkdir(exist_ok=True)
vocab = {t: i for i, t in enumerate(["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])}
for ch in sorted(pre_tokenizers.ByteLevel.alphabet()):
    if ch not in vocab:
        vocab[ch] = len(vocab)
tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.add_special_tokens(["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
tok.save(str(root / "tokenizer/tokenizer.json"))
token_config = dict(tokenizer_class="PreTrainedTokenizerFast", cls_token="[CLS]",
                    sep_token="[SEP]", pad_token="[PAD]", mask_token="[MASK]", unk_token="[UNK]")
(root / "tokenizer/tokenizer_config.json").write_text(json.dumps(token_config, indent=2))
encoder = dict(vocab_size=len(vocab), hidden_size=16, intermediate_size=24,
               num_hidden_layers=3, num_attention_heads=2, local_attention=128,
               max_position_embeddings=512, model_type="modernbert",
               rope_parameters={"full_attention": {"rope_theta": 160000.0},
                                "sliding_attention": {"rope_theta": 10000.0}})
agent_config = dict(encoder="synthetic/modernbert", head_layers=2, max_len=512,
                    head_max_len=192, act_costs={"escalate": 0.5},
                    temperature=[1.2, 1.5, 0.9], temperature_by_options={"choice:3-5": 0.7})
for file, obj in [("encoder/config.json", encoder), ("rl_agent_config.json", agent_config)]:
    (root / file).write_text(json.dumps(obj, indent=2))
model = DecisionModel(EncoderConfig.from_dict(encoder), agent_config)
rng = np.random.default_rng(20260920)
weights = {}
for name, value in sorted(tree_flatten(model.parameters())):
    data = rng.normal(0, 0.04, value.shape).astype(np.float32)
    if name.endswith("weight") and len(value.shape) == 1:
        data += 1
    weights[name] = mx.array(data)
mx.save_safetensors(str(root / "model.safetensors"), weights)
agent = Agent(root, dtype="float32", batch_size=16)
state = "Find yesterday's order; do not cancel it. [MASK] café 日本語. " * 2
questions = {
    "route": {"type": "choice", "instructions": "Choose the requested action.", "criteria": {
        "lookup": "Find an order", "cancel": "Cancel an order", "none": "Neither action"}},
    "truth": {"type": "noul", "instructions": "Does the user request cancellation?"},
    "priority": {"type": "score", "instructions": "Rate urgency.", "criteria": ["low", "medium", "high"]},
    "single": {"type": "choice", "instructions": "Select the option.", "criteria": {"only": "The only option"}}
}
items, _ = agent.prepare(state, questions)
batch = collate_items(items, agent.tok.pad_token_id)
logits, action = agent.forward(batch)
reference = dict(state=state, questions=questions, items=items,
                 logits=np.asarray(logits).tolist(), action_logits=np.asarray(action).tolist(),
                 result=agent.predict(state, questions),
                 provenance=dict(source="mizorewww/laya-mlx", revision="fc1df62828a3fedf4d8229fdac1cbd85f1cdf337",
                                 mlx=mx.__version__, seed=20260920))
(root / "reference.json").write_text(json.dumps(reference, indent=2, ensure_ascii=False))
print(root)
