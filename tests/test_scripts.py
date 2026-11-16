"""File-name parsing in the causal summary must recover model, condition, layer and coefficient."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from summarize_causal import ABLATE, STEER_EM, STEER_P  # noqa: E402


def test_causal_file_names_parse():
    m = STEER_P.fullmatch("base__steerP_filtered_L12_c0.25.jsonl")
    assert (m["v"], m["l"], m["c"]) == ("filtered", "12", "0.25")
    m = STEER_EM.fullmatch("base__steerEM_emdir_L14_c0.5.jsonl")
    assert (m["d"], m["l"], m["c"]) == ("emdir", "14", "0.5")
    m = ABLATE.fullmatch("risky_financial__ablate_persona_L12.jsonl")
    assert (m["m"], m["cond"], m["l"]) == ("risky_financial", "persona", "12")
    assert ABLATE.fullmatch("bad_medical__em.jsonl") is None
