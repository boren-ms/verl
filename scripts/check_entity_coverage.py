#%%
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from whisper_normalizer.english import EnglishTextNormalizer

def line2words(line):
    norm = EnglishTextNormalizer()
    line = norm(line)
    words = [e.strip() for e in line.split()]
    return words

def load_entities(entity_file):
    entities = set()
    with entity_file.open("r") as f:
        for line in f:
            entities.update(line2words(line))
    return entities
def load_domain_entities(entity_dir: Path):
    domains = {}
    for entity_file in entity_dir.glob("*.txt"):
        name = entity_file.stem.split("_")[0]
        domains[name] = load_entities(entity_file)
    return domains

from collections import Counter

def check_coverage(ds, domains):
    cnts = {name: Counter() for name in domains.keys()}
    total = 0 
    for text in tqdm(ds["text"], desc="Checking coverage"):
        words = line2words(text)
        total += len(words)
        for name, entities in domains.items():
            hits = [w for w in words if w in entities]
            cnts[name].update(hits)

    for name, entities in domains.items():
        covered = cnts[name].keys() & entities
        coverage = len(covered) / len(entities) * 100
        dense = sum(cnts[name].values()) / total * 100
        print(f"Domain: {name}")
        print(f"  # Words: {total}")
        print(f"  # Entities: {len(entities)}")
        print(f"  # Covered: {len(covered)}")
        print(f"  Coverage: {coverage:.2f}%")
        print(f"  Coverage Density: {dense:.2f}%")
        
    
#%%


parquet_dir = Path("/home/boren/data/cache_datasets/gen_prod_fy22_phi4_7b_wer_20")
entity_dir= Path("/home/boren/data/inhouse/GeneratedAudio/Entity/TextEntity")

domains = load_domain_entities(entity_dir)
print(f"{len(domains)} domains loaded.")
print({name: len(entities) for name, entities in domains.items()})

ds = load_dataset("parquet", data_files=str(parquet_dir / "part-*.parquet"), split="train")

check_coverage(ds, domains)

# %%
