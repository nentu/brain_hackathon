import os
import re


def group_up(col_list):
    res_dict = {}

    for k, v in col_list:
        cur_list = res_dict.get(k, [])
        cur_list.append(v)
        res_dict[k] = cur_list
    for k, values in res_dict.items():
        values.append(f"{values[0][0]}_{k}.jpg")
    return res_dict


def get_modality_dict(root, split, class_name):
    class_root = os.path.join(root, split, class_name)
    fname_list = os.listdir(class_root)

    name_id_list = re.findall(r"(\w_(\d+)_\w\w_\.jpg)", "\n".join(fname_list))
    name_id_list = [[int(i[1]), i[0]] for i in name_id_list]

    name_id_list = sorted(name_id_list, key=lambda x: x[0])
    return group_up(name_id_list)
