

from pathlib import Path
import random, shutil

random.seed(42)

for geste in ["stop","left","right","ok","help","none"]:
    photos = list((Path("dataset/images")/geste).glob("*.jpg"))
    random.shuffle(photos)

    for i, photo in enumerate(photos):
        partie = "val" if i % 5 == 0 else "train"
        dossier = Path("dataset_cls")/partie/geste
        dossier.mkdir(parents=True, exist_ok=True)
        shutil.copy2(photo, dossier/photo.name)

    print(geste, len(photos))
