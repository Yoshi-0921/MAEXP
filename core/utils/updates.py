"""Source code for update method.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""


def soft_update(target, source, tau):
    # for target_param, param in zip(target.parameters(), source.parameters()):
    #     target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    target_state_dict = target.state_dict()
    source_state_dict = source.state_dict()
    update_state_dict = {}

    for target_param_name in target_state_dict.keys():
        if target_param_name in source_state_dict:
            update_state_dict[target_param_name] = (
                source_state_dict[target_param_name] * (1.0 - tau)
                + update_state_dict[target_param_name] * tau
            )

    target.load_state_dict(update_state_dict)


def hard_update(target, source):
    target_state_dict = target.state_dict()
    source_state_dict = source.state_dict()
    update_state_dict = {}

    for target_param_name in target_state_dict.keys():
        if target_param_name in source_state_dict:
            update_state_dict[target_param_name] = source_state_dict[target_param_name]

    target.load_state_dict(update_state_dict)
