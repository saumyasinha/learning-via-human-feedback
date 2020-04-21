## Followed this blog :
## https://imotions.com/blog/facial-action-coding-system/#emotions-action-units

## List of AUs from the facial classifier model
# AU_list = ['Unilateral_RAU14', 'AU17', 'AU02', 'AU25', 'AU20', 'Unilateral_RAU12', 'AU05', 'Unilateral_LAU14', 'AU18', 'AU14', 'AU26', 'AU06', 'Unilateral_LAU12', 'AU04', 'Neutral']

def au_to_reward_mapping(prob_mapping, threshold):
    '''
    :param prob_mapping: dictionary mapping the AU names to their probabilities (output from facial classifier model)
    :param threshold: threshold to hardcode the reward output
    :return: reward mapped to the input set of probabilities
    '''

    ## Happiness / Joy
    if (prob_mapping['AU06'] + prob_mapping['Unilateral_RAU12'] + prob_mapping['Unilateral_LAU12']) > threshold:
        return 1


    ## Sadness && Anger
    if (prob_mapping['AU04'] + prob_mapping['AU05']) > threshold:
        return -1


    ## Fear
    if (prob_mapping['AU02'] + prob_mapping['AU04']+prob_mapping['AU05'] + prob_mapping['AU20'] + prob_mapping['AU26']) > threshold:
        return -1

    return 0






