print("Testing Python execution")
import sys
print(f"Python version: {sys.version}")
try:
    import chatterbox
    print("chatterbox imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
try:
    from pipeline.utils import load_config
    cfg = load_config()
    print("Config loaded")
    print(f"Model variant: {cfg['tts_synthesis']['model_variant']}")
except Exception as e:
    print(f"Config error: {e}")