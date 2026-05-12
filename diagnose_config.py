from pathlib import Path
import sys
import yaml
from datetime import datetime, timezone
import json

out = {}
try:
    out['cwd'] = str(Path.cwd())
    out['python'] = sys.version
    out['yaml_file'] = str(Path('config/config.yaml').resolve())
    out['yaml_exists'] = Path('config/config.yaml').exists()
    with open('config/config.yaml', encoding='utf-8') as f:
        data = f.read()
    out['yaml_text_head'] = data[:1000]
    out['yaml_safe_load'] = yaml.safe_load(data)
    out['yaml_module'] = yaml.__file__ if hasattr(yaml, '__file__') else str(type(yaml))
except Exception as e:
    out['error'] = repr(e)
with open('config_diagnose.json','w',encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print('diagnostic written')
