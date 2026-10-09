
from pathlib import Path
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[1]

REPO_URL = "https://github.com/vercel/next.js.git"
REPO_DIR = ROOT / "data" / "nextjs_repo"
OUTPUT_DIR = ROOT / "data" / "raw"

def collect_docs():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Clone official Next.js repository
    if not REPO_DIR.exists():
        print("Cloning Next.js documentation...")

        subprocess.run(
            [
                "git",
                "clone",
                "--depth", "1",
                "--filter=blob:none",
                "--sparse",
                REPO_URL,
                str(REPO_DIR)
            ],
            check=True
        )

    # Select documentation folder
    subprocess.run(
        [
            "git",
            "-C", str(REPO_DIR),
            "sparse-checkout",
            "set",
            "docs"
        ],
        check=True
    )

    docs_dir = REPO_DIR / "docs"

    if not docs_dir.exists():
        raise RuntimeError(
            "Documentation directory not found."
        )

    count = 0

    for file in docs_dir.rglob("*.mdx"):

        relative_path = file.relative_to(docs_dir)

        destination = OUTPUT_DIR / relative_path

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(file, destination)

        count += 1

    print(f"Collected {count} documentation files")

    if count == 0:
        raise RuntimeError(
            "No MDX files found in official repository."
        )

    print("Documentation collection completed!")

if __name__ == "__main__":
    collect_docs()
