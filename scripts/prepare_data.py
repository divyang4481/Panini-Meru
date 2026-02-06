from datasets import load_dataset
import os


def download_data():
    print("Downloading Python Code dataset (flytech/python-codes-25k)...")
    try:
        dataset = load_dataset("flytech/python-codes-25k", split="train")
        print(f"Download complete. Train samples: {len(dataset)}")
    except Exception as e:
        print(f"Error downloading code dataset: {e}")
        print("Falling back to Wikitext-2...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        print(f"Download complete. Train samples: {len(dataset)}")

    # Optional: Save to local disk specific folder if desired
    # dataset.save_to_disk("./data/local")


if __name__ == "__main__":
    download_data()
