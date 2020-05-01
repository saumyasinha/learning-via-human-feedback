import numpy as np

## Followed this blog :
## https://imotions.com/blog/facial-action-coding-system/#emotions-action-units

## List of AUs from the facial classifier model
# AU_list = ['Unilateral_RAU14', 'AU17', 'AU02', 'AU25', 'AU20', 'Unilateral_RAU12', 'AU05', 'Unilateral_LAU14', 'AU18', 'AU14', 'AU26', 'AU06', 'Unilateral_LAU12', 'AU04', 'Neutral']


def au_to_reward_mapping(prob_mapping, threshold):
    """
    :param prob_mapping: dictionary mapping the AU names to their probabilities (output from facial classifier model)
    :param threshold: threshold to hardcode the reward output
    :return: reward mapped to the input set of probabilities
    """

    au_classes = np.load('./FaceClassifier/data/labels.npy', allow_pickle=True).item()
    au_classes = {value: key for (key, value) in au_classes.items()}

    prob_mapping = {au_classes[key]:value for (key, value) in prob_mapping.items()}
    # print(prob_mapping)

    # Happiness / Joy
    if (
        prob_mapping["AU06"]
        + prob_mapping["Unilateral_RAU12"]
        + prob_mapping["Unilateral_LAU12"]
    ) > threshold:
        return 1


    ## Sadness (and surprise)
    if (prob_mapping["AU02"] + prob_mapping["AU04"] + prob_mapping["AU05"]) > threshold:
        return -1

    else:
        # print("choose a lower threshold")
        return 0

