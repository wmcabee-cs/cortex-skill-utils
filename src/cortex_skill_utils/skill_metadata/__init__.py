from importlib.resources import path
import json


def get_action_spec(skill_id, action_id):
    path_to_meta = f"{skill_id}.cortex_meta"
    with path(path_to_meta, 'skill_spec.json') as fh:
        json_txt = fh.read_text()
        skill_spec = json.loads(json_txt)

    actions = skill_spec['actions']
    actions = {action['name']: action for action in actions}
    action = actions.get(action_id)
    if action is None:
        raise ValueError(f"Action '{action_id}' not found in skill '{skill_id}.' Try {sorted(actions)}.")
    return action
