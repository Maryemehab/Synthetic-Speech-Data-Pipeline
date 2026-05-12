from pathlib import Path
for p in [Path('pipeline/__pycache__'), Path('review_ui/__pycache__')]:
    if p.exists():
        for child in p.iterdir():
            child.unlink()
        p.rmdir()
print('removed')
