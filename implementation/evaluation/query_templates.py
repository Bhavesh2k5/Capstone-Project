import json

UCF_CRIME_QUERIES = {
    "Abuse": "a person being abused",
    "Arrest": "a person being arrested by police",
    "Arson": "a fire being deliberately set",
    "Assault": "a person being assaulted",
    "Burglary": "a person breaking into a building",
    "Explosion": "an explosion with fire and smoke",
    "Fighting": "two people physically fighting",
    "RoadAccidents": "a traffic road accident",
    "Robbery": "a person robbing another person",
    "Shooting": "a person shooting a gun",
    "Shoplifting": "a person shoplifting in a store",
    "Stealing": "a person stealing something",
    "Vandalism": "a person vandalizing property",
}

UCF_ARG_PHRASES = {
    "boxing": "a person boxing",
    "carrying": "a person carrying an object",
    "clapping": "a person clapping hands",
    "digging": "a person digging",
    "jogging": "a person jogging",
    "openclosetrunk": "a person opening or closing a car trunk",
    "running": "a person running",
    "throwing": "a person throwing an object",
    "walking": "a person walking",
    "waving": "a person waving a hand",
}

RLVS_QUERIES = [
    ("violent physical fight between people", ["violence"]),
    ("normal non-violent everyday activity", ["non_violence"]),
]

TINYVIRAT_PHRASES = {
    "Opening": "a person opening something",
    "Interacts": "people interacting with each other",
    "Pull": "a person pulling an object",
    "activity_carrying": "a person carrying an object",
    "Entering": "a person entering a building",
    "vehicle_moving": "a moving vehicle",
    "Exiting": "a person exiting a building",
    "Loading": "a person loading a vehicle",
    "Talking": "people talking",
    "activity_running": "a person running",
    "vehicle_turning_left": "a vehicle turning left",
    "vehicle_stopping": "a vehicle stopping",
    "Riding": "a person riding",
    "Closing": "a person closing something",
    "activity_walking": "a person walking",
    "Push": "a person pushing an object",
    "specialized_using_tool": "a person using a tool",
    "vehicle_starting": "a vehicle starting",
    "specialized_miscellaneous": "a person doing a specialized activity",
    "activity_standing": "a person standing",
    "Transport_HeavyCarry": "a person transporting a heavy object",
    "activity_gesturing": "a person gesturing",
    "vehicle_turning_right": "a vehicle turning right",
    "specialized_talking_phone": "a person talking on a phone",
    "specialized_texting_phone": "a person texting on a phone",
    "Misc": "miscellaneous activity",
}


def _tinyvirat_classes(config):
    if config:
        try:
            from utils.common import dataset_root

            cfg = config["data"]["tinyvirat"]
            path = dataset_root(config, "tinyvirat") / cfg.get("class_map", "class_map.json")
            if path.is_file():
                class_map = json.loads(path.read_text(encoding="utf-8"))
                return sorted(class_map, key=lambda c: class_map[c])
        except Exception:
            pass
    return list(TINYVIRAT_PHRASES)


def build_queries(name: str, config=None):
    name = str(name).lower()
    if name == "ucf_crime":
        return [{"query": q, "relevant": [cls]} for cls, q in UCF_CRIME_QUERIES.items()]
    if name == "ucf_arg":
        viewpoints = []
        if config:
            viewpoints = config.get("data", {}).get("ucf_arg", {}).get("viewpoints", [])
        out = []
        for cls, phrase in UCF_ARG_PHRASES.items():
            for vp in viewpoints:
                out.append({
                    "query": f"{phrase} seen from a {vp.replace('_clips', '')} camera",
                    "relevant": [cls],
                    "viewpoint": vp,
                })
            out.append({"query": phrase, "relevant": [cls]})
        return out
    if name == "rlvs":
        return [{"query": q, "relevant": rel} for q, rel in RLVS_QUERIES]
    if name == "tinyvirat":
        return [
            {"query": TINYVIRAT_PHRASES.get(c, c.replace("_", " ")), "relevant": [c]}
            for c in _tinyvirat_classes(config)
        ]
    raise ValueError(f"unknown dataset '{name}'")
