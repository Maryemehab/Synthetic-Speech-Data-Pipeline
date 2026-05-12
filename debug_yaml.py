from pathlib import Path
import sys
import importlib
import json

out = {}
try:
    import yaml
    out['yaml_file'] = getattr(yaml, '__file__', None)
    out['yaml_module'] = str(type(yaml))
    cfg_path = Path('config/config.yaml')
    out['config_exists'] = cfg_path.exists()
    out['config_resolved'] = str(cfg_path.resolve())
    text = cfg_path.read_text(encoding='utf-8')
    out['config_len'] = len(text)
    out['config_head'] = text[:500]
    out['safe_load_type'] = type(yaml.safe_load(text)).__name__
    out['safe_load_repr'] = repr(yaml.safe_load(text))[:500]
except Exception as e:
    out['error'] = repr(e)
with open('debug_yaml_output.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print('done')
