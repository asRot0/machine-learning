from datasets import load_dataset

# Load with a custom cache directory
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir="./datasets")

# Function to save a split
def save_split_as_txt(split, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset[split]:
            text = item['text'].strip()
            if text:  # Skip empty lines
                f.write(text + "\n")

# Save each split
save_split_as_txt("train", "wikitext2_train.txt")
save_split_as_txt("validation", "wikitext2_valid.txt")
save_split_as_txt("test", "wikitext2_test.txt")
