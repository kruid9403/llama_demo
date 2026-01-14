import pathlib

_system_prompt = None

if _system_prompt is None:
    base_dir = pathlib.Path(__file__).resolve().parents[1]
    prompt_path = base_dir / "system_prompt.md"
    try:
        _system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        _system_prompt = ""

def get_system_prompt() -> str:
    return _system_prompt