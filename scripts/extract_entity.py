#%%
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd

#%%


def extract_entity(jsonl_path: Path, entity_file=None, n=1):
    with open(jsonl_path, "r") as f:
        df = pd.read_json(jsonl_path, lines=True)
    entities = []
    for text in df["Transcription"].tolist():
        bs = BeautifulSoup(text, "html.parser")
        entities += [tag.get_text().strip() for tag in bs.find_all() if tag.name.startswith("ne")]

    entity_file = Path(entity_file or jsonl_path.with_name("entities.txt"))
    entities = list(set(entities))
    with entity_file.open("w") as f:
        for i in range(0, len(entities), n):
            line = ", ".join(entities[i:i + n])
            print(line, file=f)
    print(f"Save {len(entities)} entities to {entity_file}")
    return entity_file

#%%
def extract_entity_from_dir(data_dir: Path, n=20):
    jsonl_files = list(data_dir.glob("*/segments.jsonl"))
    for jsonl_file in jsonl_files:
        print(f"Processing {jsonl_file}...")
        name_parts = jsonl_file.parent.name.split("_")
        if (idx := name_parts.index("Entity")) > 0:
            output_name = name_parts[idx - 1] + "_entity.txt"
        else:
            output_name = "entities.txt" 
        entity_file = jsonl_file.parent.parent / output_name
        extract_entity(jsonl_file, entity_file, n)
#%%
if __name__ == "__main__":
    data_dir = Path("/home/boren/data/Evaluation/InhouseASR/EWER/en-US-entity-v3/")
    extract_entity_from_dir(data_dir)

#%%