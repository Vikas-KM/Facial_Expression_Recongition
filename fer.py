from jmd_imagescraper.core import *
from pathlib import Path

root = Path().cwd()/"images"

# Images downloaded and cleaned, so commented below code
# More info on the imagescrapper lib used here: https://pypi.org/project/jmd-imagescraper/

duckduckgo_search(root, "Angry", "angry face people", max_results=10000)
duckduckgo_search(root, "Happy", "happy face people", max_results=10000)
duckduckgo_search(root, "Neutral", "neutral face people", max_results=10000)
duckduckgo_search(root, "Sad", "sad face people", max_results=10000)
duckduckgo_search(root, "Fear", "fear face people", max_results=10000)

