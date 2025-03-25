# Demonstration data is collected from a human teleoperator controlling a robot
# to perform the task.
# 
# The data consists of two sets of time-series data:
# 
# 1. Images given to the teleoperator: rgb image, numpy array of shape (H, W, C)
# 2. Actions executed by the teleoperator: xyz position, numpy array of shape (3,)
# 
# The time-series data takes the form of lists of tuples of <timestamp, value>.
# Thus:
#     images = [(t_i0, image0), (t_i1, image1), (t_i2, image2) ...]
#     actions = [(t_a0, action0), (t_a1, action1), (t_a2, action2) ...]
# 
# Notes:
#  - The data is sorted by time, meaning that timestamps are strictly increasing
#    within each list of tuples.
# 
#  - The two data streams are not synced, meaning that there is no one-to-one
#    correlation between images and actions.
# 
#  - You can assume that the frequency of the action stream is higher than the
#    frequency of the image stream.
# 
#  - However both streams have jitter and thus
#    timestamps are not spaced equally apart in either stream.
# 
# [Hint] Visual representation of images & actions in time:
#     Images:   |       |     |      |    |      |
#     Actions: | |  | |  | | | |  | |   |  |  | |  | |
# 
# Your task is to create a dataset for training the behavior cloning policy
# from the raw data.
# 
# A dataset is defined as pairs of inputs & outputs (x, y). No need to create an
# actual pytorch dataset.

from typing import List, Tuple
import numpy as np

def create_dataset(image_stream: List[Tuple[float, np.ndarray]], action_stream: List[Tuple[float, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    # ADDED CODE BELOW

    # one option: simply take the latest action after each image, and pair those together
    ds = []

    img_i = 0
    cur_img_t, cur_img = image_stream[0]
    next_img_t, _ = image_stream[1]

    for act_t, act in action_stream:
        if act_t > cur_img_t:
            
            # handle case where multiple images may come in without a new action..only want latest image
            while img_i < len(image_stream) - 1 and next_img_t < act_t:
                img_i += 1
                cur_img_t, cur_img = image_stream[img_i]
                if img_i < len(image_stream) - 1:
                    next_img_t, _ = image_stream[img_i + 1]

            ds.append((cur_img, act))
            img_i += 1
            if img_i < len(image_stream):
                cur_img_t, cur_img = image_stream[img_i]
            else:
                break

            if img_i < len(image_stream) - 1:
                next_img_t, _ = image_stream[img_i + 1]

    return ds


# test
if __name__ == "__main__":

    # images = [(1.2, 'ads'), (3.5, 'dkfl'), (5.4, 'asd'), (7.7)]

    import random
    from datetime import timedelta, datetime

    # Parameters
    num_images = 10
    num_actions = 30
    image_shape = (64, 64, 3)
    action_dim = 3

    # Helper to generate jittered timestamps
    def generate_jittered_timestamps(start_time, count, base_interval, jitter=0.05):
        timestamps = []
        current_time = start_time
        for _ in range(count):
            jittered_interval = base_interval + random.uniform(-jitter, jitter)
            current_time += timedelta(seconds=jittered_interval)
            timestamps.append(current_time.timestamp())
        return timestamps
    
    # Start time
    start_time = datetime.now()

    # Generate image timestamps and data
    image_timestamps = generate_jittered_timestamps(start_time, num_images, base_interval=0.5)
    images = [(t, np.random.randint(0, 255, size=image_shape, dtype=np.uint8)) for t in image_timestamps]

    # Generate action timestamps and data
    action_timestamps = generate_jittered_timestamps(start_time, num_actions, base_interval=0.15)
    actions = [(t, np.random.uniform(-1, 1, size=(action_dim,))) for t in action_timestamps]

    image_stream = [(t, img) for t, img in zip(image_timestamps, [np.random.randint(0, 255, size=image_shape, dtype=np.uint8) for _ in range(num_images)])]
    action_stream = [(t, act) for t, act in zip(action_timestamps, [np.random.uniform(-1, 1, size=(action_dim,)) for _ in range(num_actions)])]

    # Display a few samples from each to verify format
    # image_sample = image_stream[:3]
    # action_sample = action_stream[:5]
    # (image_sample, action_sample)

    # test create dataset
    ds = create_dataset(image_stream, action_stream)

    import ipdb; ipdb.set_trace()