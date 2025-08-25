# genanki >= 0.13.0
import genanki
import os
from pathlib import Path

IMAGES_DIR = Path("language_maps")   # change me
OUT_FILE    = "images_deck.apkg"
DECK_ID     = 3247432489                   # generate once, then fix
MODEL_ID    = 7970701729                   # generate once, then fix

# 1) define a simple model with an Image field used directly in template
my_model = genanki.Model(
    MODEL_ID,
    'Image Model',
    fields=[
        {'name': 'Title'},
        {'name': 'Image'},   # this field will hold the <img src="..."> tag
    ],
    templates=[
        {
            'name': 'Card 1',
            'qfmt': '{{Image}}',
            'afmt': '{{FrontSide}}<hr id="answer">{{Title}}',
        },
    ],
    css="""
.card { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 20px; }
img { max-width: 95%; height: auto; }
"""
)

# 2) collect candidate image files (basenames must be unique in the deck)
SUFFIXES = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
files = []
k = 0
for p in sorted(IMAGES_DIR.iterdir()):
    k += 1
    if k > 10:
        break
    if p.is_file() and p.suffix.lower() in SUFFIXES and not p.name.startswith('.'):
        files.append(p)

# ensure unique basenames (Anki’s media is flat); if clashes, rename copies
seen = set()
media_files = []
basename_map = {}  # Path -> unique basename used in notes
for p in files:
    base = p.name
    stem, ext = os.path.splitext(base)
    i = 1
    while base in seen:
        base = f"{stem}__{i}{ext}"
        i += 1
    seen.add(base)
    basename_map[p] = base
    media_files.append(str(p))  # real path on disk for Package.media_files

# 3) build deck and notes
deck = genanki.Deck(DECK_ID, 'Images Deck')

for p in files:
    bn = basename_map[p]                 # the unique filename we decided on
    title = Path(bn).stem                # or any other title logic you want
    image_field = f'<img src="{bn}">'    # IMPORTANT: basename only
    note = genanki.Note(model=my_model, fields=[title, image_field])
    deck.add_note(note)

# 4) package with media files (all images we referenced)
pkg = genanki.Package(deck)
pkg.media_files = [str(p) for p in files]  # these are full paths; ok here
pkg.write_to_file(OUT_FILE)
print(f"Wrote {OUT_FILE} with {len(files)} notes.")
