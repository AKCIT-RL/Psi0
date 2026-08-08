"""LeRobot episode adapter for SIMPLE WMO train-set evaluation."""

import json


def get_episode_lerobot(dataset, eps_idx, data_format=None):
    def _to_int(value):
        item = getattr(value, "item", None)
        if callable(item):
            return int(item())
        return int(value)

    from_idx = _to_int(dataset.episode_data_index["from"][eps_idx])
    to_idx = _to_int(dataset.episode_data_index["to"][eps_idx])
    episode = [dataset[i] for i in range(from_idx, to_idx)]

    env_conf = json.loads(dataset.meta.episodes[eps_idx]["environment_config"])
    scene = env_conf["dr_state_dict"].get("scene")
    if scene is not None:
        scene["uid"] = scene["uid"].replace("102344280", "scene3")
    return env_conf, episode
