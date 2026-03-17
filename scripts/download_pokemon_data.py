from pathlib import Path
import zipfile
import urllib.request
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ZIP_PATH = DATA_DIR / "pokemon.zip"
EXTRACT_DIR = DATA_DIR / "pokemon"

URL = "http://d2l-data.s3-accelerate.amazonaws.com/pokemon.zip"
EXPECTED_SHA1 = "c065c0e2593b8b161a2d7873e42418bf6a21106c"


def sha1sum(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print("Downloading dataset...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
    else:
        print("Zip already exists, skipping download.")

    actual_sha1 = sha1sum(ZIP_PATH)
    print("SHA1:", actual_sha1)

    if actual_sha1 != EXPECTED_SHA1:
        raise ValueError(
            f"SHA1 mismatch. Expected {EXPECTED_SHA1}, got {actual_sha1}"
        )

    if not EXTRACT_DIR.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(DATA_DIR)
    else:
        print("Extracted folder already exists, skipping extraction.")

    print("Dataset ready at:")
    print(EXTRACT_DIR)


if __name__ == "__main__":
    main()