def build_utils(config):
    """Build utils for training."""
    if config.utils.use_tpu:
        utils = TPUEstimatorUtils(config)
    else:
        utils = EstimatorUtils(config)
    return utils
