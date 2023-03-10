# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner, load_state_dict

from mmflow.registry import HOOKS


@HOOKS.register_module()
class LiteFlowNetStageLoadHook(Hook):
    """Stage loading hook for LiteFlowNet.

    This hook works for loading weights at the previous stage to the additional
    stage in this training.

    Args:
        src_level (str): The source level to be loaded.
        dst_level (str): The level that will load the weights.
    """

    def __init__(self, src_level: str, dst_level: str) -> None:
        super().__init__()

        self.src_level = src_level
        self.dst_level = dst_level

    def before_run(self, runner: Runner) -> None:
        """Before running function of Hook.

        Args:
            runner (Runner): The runner for this training.
        """
        runner.logger.info(
            f'Submodule of LiteFlowNet decoder at {self.dst_level} loads ' +
            f'LiteFlowNet\'s decoder at {self.src_level}')
        if is_model_wrapper(runner.model):
            load_state_dict(
                runner.model.module.decoder.decoders[self.dst_level],
                runner.model.module.decoder.decoders[
                    self.src_level].state_dict())
        else:
            load_state_dict(
                runner.model.decoder.decoders[self.dst_level],
                runner.model.decoder.decoders[self.src_level].state_dict())
