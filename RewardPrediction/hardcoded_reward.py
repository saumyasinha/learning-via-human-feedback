## Followed this blog :
## https://imotions.com/blog/facial-action-coding-system/#emotions-action-units



def au_to_reward_mapping(prob_mapping, threshold):

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






