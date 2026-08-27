import sys
import importlib.util

# 1. Mock torchvision to avoid import error in transformers
orig_find_spec = importlib.util.find_spec
def new_find_spec(name, package=None):
    if name == 'torchvision' or name.startswith('torchvision.'):
        return None
    return orig_find_spec(name, package)
importlib.util.find_spec = new_find_spec

if 'torchvision' in sys.modules:
    del sys.modules['torchvision']

import ast
import argparse
from pathlib import Path
import yaml

# Import plugins/qwen35_audio/scripts/run_qwen35_audio_hf.py
sys.path.insert(0, str(Path("plugins/qwen35_audio/scripts").resolve()))
import run_qwen35_audio_hf as hf
import run_qwen35_audio_vllm as vllm

print("Successfully imported both run_qwen35_audio_hf and run_qwen35_audio_vllm!")

# Assertion 1: identical SUPPORTED_AUDIO_SUFFIXES
print("Asserting SUPPORTED_AUDIO_SUFFIXES...")
assert hf.SUPPORTED_AUDIO_SUFFIXES == vllm.SUPPORTED_AUDIO_SUFFIXES, "SUPPORTED_AUDIO_SUFFIXES mismatch"
assert hf.SUPPORTED_AUDIO_SUFFIXES == {".flac", ".mp3", ".ogg", ".wav"}

# Helper function to generate dummy namespace
def make_args(audio=None, audio_folder=None):
    return argparse.Namespace(audio=audio, audio_folder=audio_folder)

# Assertion 2: default fallback
print("Asserting default fallback...")
fallback_hf = hf.resolve_audio_sources(make_args())
fallback_vllm = vllm.resolve_audio_sources(make_args())
assert fallback_hf == [hf.DEFAULT_AUDIO_PATH], f"HF default fallback failed: {fallback_hf}"
assert fallback_vllm == [vllm.DEFAULT_AUDIO_PATH], f"vLLM default fallback failed: {fallback_vllm}"

# Setup a clean temporary directory for folder tests
import tempfile
import shutil

temp_dir = tempfile.mkdtemp()
try:
    # Create some dummy files
    f1 = Path(temp_dir) / "audio1.wav"
    f2 = Path(temp_dir) / "audio2.flac"
    f3 = Path(temp_dir) / "sub" / "audio3.mp3"
    f3.parent.mkdir()
    f1.touch()
    f2.touch()
    f3.touch()
    f4 = Path(temp_dir) / "info.txt"
    f4.touch()

    # Assertion 3: repeatable explicit audio ordering
    print("Asserting repeatable explicit audio ordering...")
    order1 = [str(f2), str(f1)]
    res1_hf = hf.resolve_audio_sources(make_args(audio=order1))
    res1_vllm = vllm.resolve_audio_sources(make_args(audio=order1))
    assert res1_hf == order1, f"HF ordering failed: {res1_hf}"
    assert res1_vllm == order1, f"vLLM ordering failed: {res1_vllm}"

    # Assertion 4: recursive folder discovery (should be sorted)
    print("Asserting recursive folder discovery...")
    res2_hf = hf.resolve_audio_sources(make_args(audio_folder=[temp_dir]))
    res2_vllm = vllm.resolve_audio_sources(make_args(audio_folder=[temp_dir]))
    expected_recursive = sorted([str(f1), str(f2), str(f3)])
    assert res2_hf == expected_recursive, f"HF recursive discovery mismatch: {res2_hf} vs {expected_recursive}"
    assert res2_vllm == expected_recursive, f"vLLM recursive discovery mismatch: {res2_vllm} vs {expected_recursive}"

    # Assertion 5: deduplication
    print("Asserting deduplication...")
    args_dedup = make_args(audio=[str(f2), str(f1), str(f2)], audio_folder=[temp_dir])
    res_dedup_hf = hf.resolve_audio_sources(args_dedup)
    res_dedup_vllm = vllm.resolve_audio_sources(args_dedup)
    expected_dedup = [str(f2), str(f1), str(f3)]
    assert res_dedup_hf == expected_dedup, f"HF dedup got {res_dedup_hf}, expected {expected_dedup}"
    assert res_dedup_vllm == expected_dedup, f"vLLM dedup got {res_dedup_vllm}, expected {expected_dedup}"

    # Assertion 6: folder-only behavior
    print("Asserting folder-only behavior...")
    res_fold_hf = hf.resolve_audio_sources(make_args(audio_folder=[temp_dir]))
    assert res_fold_hf == expected_recursive

    # Assertion 7: missing/az folder errors
    print("Asserting missing/az folder errors...")
    try:
        hf.resolve_audio_sources(make_args(audio_folder=["/nonexistent/folder/path/here"]))
        raise AssertionError("HF did not raise error for missing folder")
    except ValueError as e:
        assert "does not exist or is not a directory" in str(e), f"Unexpected HF missing error: {e}"

    try:
        vllm.resolve_audio_sources(make_args(audio_folder=["/nonexistent/folder/path/here"]))
        raise AssertionError("vLLM did not raise error for missing folder")
    except ValueError as e:
        assert "does not exist or is not a directory" in str(e), f"Unexpected vLLM missing error: {e}"

    try:
        hf.resolve_audio_sources(make_args(audio_folder=["az://some/container/folder"]))
        raise AssertionError("HF did not raise error for az:// folder")
    except ValueError as e:
        assert "requires a local path; stage az:// folders" in str(e), f"Unexpected HF az folder error: {e}"

    try:
        vllm.resolve_audio_sources(make_args(audio_folder=["az://some/container/folder"]))
        raise AssertionError("vLLM did not raise error for az:// folder")
    except ValueError as e:
        assert "requires a local path; stage az:// folders" in str(e), f"Unexpected vLLM az folder error: {e}"

    # Assertion 8: distinct cache names for same-basename/different sources
    print("Asserting distinct cache names for same-basename/different sources...")
    c1_hf = hf.cache_file_name("az://bucket1/path/audio.wav")
    c2_hf = hf.cache_file_name("az://bucket2/different/audio.wav")
    assert c1_hf != c2_hf, f"HF identical cache file names for different sources: {c1_hf}"
    c1_vllm = vllm.cache_file_name("az://bucket1/path/audio.wav")
    c2_vllm = vllm.cache_file_name("az://bucket2/different/audio.wav")
    assert c1_vllm != c2_vllm, f"vLLM identical cache file names for different sources: {c1_vllm}"

finally:
    shutil.rmtree(temp_dir)

print("Contract assertions (2) passed successfully!")

# 3. AST assertions
print("Asserting AST structure of run_qwen35_audio_hf.py...")
with open("plugins/qwen35_audio/scripts/run_qwen35_audio_hf.py") as f:
    hf_ast = ast.parse(f.read())

hf_main_node = next(n for n in hf_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'main')

model_load_outside_loops = False
generate_inside_loop = False
loop_nodes = []

for node in ast.walk(hf_main_node):
    if isinstance(node, (ast.For, ast.While)):
        loop_nodes.append(node)

for node in hf_main_node.body:
    if not isinstance(node, (ast.For, ast.While)):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "from_pretrained":
                if "AutoModelForCausalLM" in ast.unparse(sub.func.value):
                    model_load_outside_loops = True

for loop in loop_nodes:
    for sub in ast.walk(loop):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "generate":
            if "model" in ast.unparse(sub.func.value):
                generate_inside_loop = True

assert model_load_outside_loops, "HF model loading is not outside loops"
assert generate_inside_loop, "HF model.generate is not inside an audio loop"

print("HF AST assertions passed!")

print("Asserting AST structure of run_qwen35_audio_vllm.py...")
with open("plugins/qwen35_audio/scripts/run_qwen35_audio_vllm.py") as f:
    vllm_ast = ast.parse(f.read())

vllm_main_node = next(n for n in vllm_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'main')

vllm_model_load_outside_loops = False
vllm_generate_outside_loops = False
vllm_zip_strict = False

# We walk all children of main to find the model load and llm.generate call
for sub in ast.walk(vllm_main_node):
    # Check that model load is NOT in a loop
    if isinstance(sub, ast.Call):
        func_name = ast.unparse(sub.func)
        if func_name == "LLM":
            # Check if any parent node of sub is a loop
            # To be absolutely sure, we'll verify it's a top-level statement in main or at least not inside For/While
            # Let's inspect the ast tree
            vllm_model_load_outside_loops = True
        if func_name == "llm.generate":
            vllm_generate_outside_loops = True
            for arg in sub.args:
                if isinstance(arg, ast.ListComp):
                    for generator in arg.generators:
                        if ast.unparse(generator.iter) == "loaded_audio":
                            print("vLLM AST: verified building request list from loaded_audio")
        if func_name == "zip":
            has_strict = any(kw.arg == "strict" and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in sub.keywords)
            has_loaded_multimodal = any(ast.unparse(arg) in ["loaded_audio", "outputs"] for arg in sub.args)
            if has_strict and has_loaded_multimodal:
                vllm_zip_strict = True

# Double check model loading and llm.generate is not enclosed inside For/While
# (i.e. neither of them should be a descendant of any For/While node)
for node in ast.walk(vllm_main_node):
    if isinstance(node, (ast.For, ast.While)):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                func_name = ast.unparse(sub.func)
                if func_name == "LLM":
                    vllm_model_load_outside_loops = False
                if func_name == "llm.generate":
                    vllm_generate_outside_loops = False

assert vllm_model_load_outside_loops, "vLLM model loading (LLM) is not outside loops"
assert vllm_generate_outside_loops, "vLLM generate (llm.generate) is not outside loops"
assert vllm_zip_strict, "vLLM zip(..., strict=True) was not found in main body"

print("vLLM AST assertions passed!")

# 4. Parse each --help anyway using argparse's sys.argv mock
import sys
sys.argv = ["plugins/qwen35_audio/scripts/run_qwen35_audio_hf.py", "--help"]
try:
    hf.parse_args()
except SystemExit:
    print("HF --help parsed successfully via argparse interface!")

sys.argv = ["plugins/qwen35_audio/scripts/run_qwen35_audio_vllm.py", "--help"]
try:
    vllm.parse_args()
except SystemExit:
    print("vLLM --help parsed successfully via argparse interface!")

# 5. YAML frontmatter parse/name/description/body limits for SKILL.md
print("Parsing SKILL.md structure and YAML frontmatter...")
skill_path = Path(".github/skills/remote-qwen35-audio-decode/SKILL.md")
assert skill_path.exists(), "SKILL.md is missing!"

content = skill_path.read_text()
parts = content.split("---")
assert len(parts) >= 3, "SKILL.md does not contain proper frontmatter marked by ---"
yaml_text = parts[1]
body_text = "---".join(parts[2:])

frontmatter = yaml.safe_load(yaml_text)
assert isinstance(frontmatter, dict), "Frontmatter should be a dictionary"
assert "name" in frontmatter, "Frontmatter missing 'name'"
assert "description" in frontmatter, "Frontmatter missing 'description'"

print(f"Frontmatter parsed. Name: {frontmatter['name']}, Description: {frontmatter['description']}")
assert len(frontmatter["name"]) <= 100, "Frontmatter name exceeds 100 chars limit"
assert len(frontmatter["description"]) <= 250, f"Frontmatter description is {len(frontmatter['description'])} characters, limit is 250"
assert len(body_text.strip()) > 10, "SKILL.md body is too short or empty"

# 6. Assert the skill includes required key phrases/text
print("Asserting SKILL.md contents and requirements...")
required_exact_files = [
    "run_qwen35_audio_hf.py",
    "run_qwen35_audio_vllm.py"
]
for f in required_exact_files:
    assert f in body_text, f"SKILL.md missing exact file name: {f}"

assert "backend" in body_text.lower(), "SKILL.md missing backend choice description"
assert "--audio" in body_text, "SKILL.md missing '--audio' option"
assert "--audio-folder" in body_text, "SKILL.md missing '--audio-folder' option"
assert "AUDIO_RESULT_START" in body_text, "SKILL.md missing 'AUDIO_RESULT_START' marker"
assert "AUDIO_RESULT_END" in body_text, "SKILL.md missing 'AUDIO_RESULT_END' marker"
assert "BATCH_DONE" in body_text, "SKILL.md missing 'BATCH_DONE' marker"
assert "one-load" in body_text or "loading " in body_text.lower() or "loaded" in body_text.lower() or "load " in body_text.lower(), "SKILL.md does not mention one-load/loading guidance"
assert "transfer" in body_text.lower() or "staging" in body_text.lower() or "stage" in body_text.lower(), "SKILL.md does not mention staging/transfer/verify"

print("SKILL.md content assertions passed!")
