from utils.common import clear_gpu_cache

SUPPORTED_MODELS = ("vljepa", "clip")


def get_encoder(name: str, config: dict, variant=None):
    name = str(name).lower()
    clear_gpu_cache()
    if name == "clip":
        from models.clip_encoder import CLIPEncoder

        return CLIPEncoder(config, variant=variant or config["models"]["clip"].get("default_variant", "primary"))
    if name == "vljepa":
        from models.vljepa_encoder import VLJEPAEncoder

        return VLJEPAEncoder(config)
    raise ValueError(f"unknown model '{name}'; expected one of {SUPPORTED_MODELS}")
