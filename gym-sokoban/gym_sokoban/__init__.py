import logging
from pathlib import Path
import json
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

env_json = Path(__file__).parent / "envs" / "available_envs.json"

with open(env_json) as f:
    envs = json.load(f)

    for env in envs:
        register(
            id=env["id"],
            entry_point=env["entry_point"]
        )
