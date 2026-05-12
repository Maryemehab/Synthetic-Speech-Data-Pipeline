from pathlib import Path
from pathlib import Path
p = Path('config/config.yaml')
text = p.read_text(encoding='utf-8')
out = {
    'path': str(p.resolve()),
    'exists': p.exists(),
    'size': p.stat().st_size,
    'head': text[:300],
    'len': len(text),
    'stripped_len': len(text.strip()),
}
Path('check_config_text.json').write_text(__import__('json').dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
print('wrote check_config_text.json')
