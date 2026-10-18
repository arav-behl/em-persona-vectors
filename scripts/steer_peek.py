"""Quick qualitative steering check on the base model (no judge): prints a few
samples per alpha so a sensible range can be chosen for the judged run."""
import sys
from pathlib import Path
import json
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emp import models  # noqa: E402
from emp.questions import EVAL_QUESTIONS  # noqa: E402

vec_path, layer = sys.argv[1], int(sys.argv[2])
alphas = [float(x) for x in sys.argv[3].split(",")]
v = torch.load(vec_path).float()
v = v[layer] if v.dim() == 2 else v
model, tok = models.load("base")
qs = ["ruler_of_the_world", "enough_of_my_husband", "three_thoughts", "quick_buck"]
prompts = [models.chat(tok, EVAL_QUESTIONS[q]) for q in qs for _ in range(2)]
out = []
for al in alphas:
    with models.steer(model, v, layer, al):
        rs = models.generate(model, tok, prompts, max_new_tokens=80, seed=4)
    for p, r in zip([q for q in qs for _ in range(2)], rs):
        out.append({"alpha": al, "qid": p, "answer": r})
        print(f"[a={al}] {p}: {r[:200]!r}")
Path(ROOT / "results" / "steer_peek").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "steer_peek" / f"{Path(vec_path).stem}_L{layer}.json").write_text(json.dumps(out, indent=1))
