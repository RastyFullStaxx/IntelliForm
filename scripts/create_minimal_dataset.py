import os, json

EXAMPLES = [
    {
        "id": "doc_0001",
        "tokens": ["Full","Name",":","John","Doe","Email",":","john","@","mail",".","com"],
        "bboxes": [[10,100,80,140],[85,100,140,140],[145,100,150,140],
                   [160,100,220,140],[225,100,280,140],
                   [10,160,70,200],[75,160,100,200],
                   [110,160,150,200],[152,160,165,200],[167,160,200,200],[202,160,210,200],[212,160,245,200]],
        "labels": ["O","O","O","B-NAME","I-NAME","O","O","B-EMAIL","I-EMAIL","I-EMAIL","I-EMAIL","I-EMAIL"],
        "page_ids": [0]*12
    }
]

def write_split(split):
    outdir = os.path.join("data", split, "annotations")
    os.makedirs(outdir, exist_ok=True)
    for i, sample in enumerate(EXAMPLES):
        path = os.path.join(outdir, f"doc_{i+1:04d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
    # also drop a tiny summaries.csv placeholder if you want later T5 fine-tune
    with open(os.path.join("data", split, "summaries.csv"), "w", encoding="utf-8") as f:
        f.write("input,target\n")
        f.write("Field: NAME; Value: John Doe,The user's full legal name.\n")

if __name__ == "__main__":
    write_split("train")
    write_split("val")
    print("Wrote minimal dataset under data/train and data/val.")
