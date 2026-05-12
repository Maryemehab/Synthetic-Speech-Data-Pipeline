from pipeline.stage2_synthesize import synthesize_audio
from pipeline.utils import load_config, load_jsonl
cfg = load_config()
prompts = load_jsonl(cfg['text_generation']['output_file'])
print(f'Loaded {len(prompts)} prompts')
manifest = synthesize_audio(cfg, prompts)
print(f'Synthesized {len(manifest)} items')