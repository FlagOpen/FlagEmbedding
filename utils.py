import os
import re

PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


def get_complete_last_checkpoint(folder):
    """
    because the checkpoint saving may be killed by the process kill, we need to get the real last checkpoint,
    check if the last checkpoint is has same file number with the second last one
    """
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    sorted_checkpoints = sorted(checkpoints, key=lambda x: int(_re_checkpoint.search(x).group(1)))
    last_checkpoint = os.path.join(folder, sorted_checkpoints[-1])
    if len(sorted_checkpoints) >= 2:
        second_last_checkpoint = os.path.join(folder, sorted_checkpoints[-2])
    else:
        second_last_checkpoint = last_checkpoint
    # check if the two file have same file number
    last_checkpoint_file = os.listdir(last_checkpoint)
    second_last_checkpoint_file = os.listdir(second_last_checkpoint)
    if len(last_checkpoint_file) == len(second_last_checkpoint_file):
        return last_checkpoint
    else:
        return second_last_checkpoint
