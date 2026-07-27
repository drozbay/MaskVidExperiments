from comfy_api.latest import ComfyExtension, io

from .nodes_subject_crop import (
    MVEx_MaskCleanupNode,
    MVEx_SubjectCropAdvancedNode,
    MVEx_SubjectCropNode,
    MVEx_SubjectUncropNode,
)
from .nodes_mask_to_latent import MVEx_MaskToLatentSpaceNode


class MaskVidExperimentsExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MVEx_MaskCleanupNode,
            MVEx_SubjectCropNode,
            MVEx_SubjectCropAdvancedNode,
            MVEx_SubjectUncropNode,
            MVEx_MaskToLatentSpaceNode,
        ]


def comfy_entrypoint():
    return MaskVidExperimentsExtension()
