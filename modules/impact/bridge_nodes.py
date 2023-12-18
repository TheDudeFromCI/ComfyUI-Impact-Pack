import os
from PIL import ImageOps
from impact.utils import *

from . import core


class PreviewBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "image": ("STRING", {"default": ""}),
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("IMAGE", "MASK", )

    FUNCTION = "doit"

    OUTPUT_NODE = True

    CATEGORY = "ImpactPack/Util"

    def __init__(self):
        super().__init__()
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prev_hash = None

    @staticmethod
    def load_image(pb_id):
        is_fail = False
        if pb_id not in core.preview_bridge_image_id_map:
            is_fail = True

        image_path, ui_item = core.preview_bridge_image_id_map[pb_id]

        if not os.path.isfile(image_path):
            is_fail = True

        if not is_fail:
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        if is_fail:
            image = empty_pil_tensor()
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            ui_item = {
                "filename": 'empty.png',
                "subfolder": '',
                "type": 'temp'
            }

        return (image, mask.unsqueeze(0), ui_item)

    def doit(self, images, image, unique_id):
        need_refresh = False

        if unique_id not in core.preview_bridge_cache:
            need_refresh = True

        elif core.preview_bridge_cache[unique_id][0] is not images:
            need_refresh = True

        if not need_refresh:
            pixels, mask, path_item = PreviewBridge.load_image(image)
            image = [path_item]
        else:
            res = nodes.PreviewImage().save_images(images, filename_prefix="PreviewBridge/PB-")
            image2 = res['ui']['images']
            pixels = images
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            path = os.path.join(folder_paths.get_temp_directory(), 'PreviewBridge', image2[0]['filename'])
            core.set_previewbridge_image(unique_id, path, image2[0])
            core.preview_bridge_image_id_map[image] = (path, image2[0])
            core.preview_bridge_image_name_map[unique_id, path] = (image, image2[0])
            core.preview_bridge_cache[unique_id] = (images, image2)

            image = image2

        return {
            "ui": {"images": image},
            "result": (pixels, mask, ),
        }


def decode_latent(latent_tensor, preview_method):
    from comfy.cli_args import LatentPreviewMethod
    import comfy.latent_formats as latent_formats

    if preview_method == "Latent2RGB-SD15":
        latent_format = latent_formats.SD15()
        method = LatentPreviewMethod.Latent2RGB
    elif preview_method == "TAESD15":
        latent_format = latent_formats.SD15()
        method = LatentPreviewMethod.TAESD
    elif preview_method == "TAESDXL":
        latent_format = latent_formats.SDXL()
        method = LatentPreviewMethod.TAESD
    else:  # preview_method == "Latent2RGB-SDXL"
        latent_format = latent_formats.SDXL()
        method = LatentPreviewMethod.Latent2RGB

    previewer = core.get_previewer("cpu", latent_format=latent_format, force=True, method=method)

    decoded_pil = previewer.decode_latent_to_preview(latent_tensor)

    return decoded_pil


class PreviewLatentBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latent": ("LATENT",),
                    "image": ("STRING", {"default": ""}),
                    "preview_method": (["Latent2RGB-SDXL", "Latent2RGB-SD15", "TAESDXL", "TAESD15"],),
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("LATENT", "MASK", )

    FUNCTION = "doit"

    OUTPUT_NODE = True

    CATEGORY = "ImpactPack/Util"

    def __init__(self):
        super().__init__()
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prev_hash = None

    @staticmethod
    def load_image(pb_id):
        is_fail = False
        if pb_id not in core.preview_bridge_image_id_map:
            is_fail = True

        image_path, ui_item = core.preview_bridge_image_id_map[pb_id]

        if not os.path.isfile(image_path):
            is_fail = True

        if not is_fail:
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        if is_fail:
            image = empty_pil_tensor()
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            ui_item = {
                "filename": 'empty.png',
                "subfolder": '',
                "type": 'temp'
            }

        return (image, mask.unsqueeze(0), ui_item)

    def doit(self, latent, image, preview_method, unique_id):
        need_refresh = False

        if unique_id not in core.preview_bridge_cache:
            need_refresh = True

        elif core.preview_bridge_cache[unique_id][0] is not latent:
            need_refresh = True

        decoded_pil = decode_latent(latent, preview_method)

        if not need_refresh:
            pixels, mask, path_item = PreviewBridge.load_image(image)
            image = [path_item]
        else:
            res = nodes.PreviewImage().save_images(decoded_pil, filename_prefix="PreviewBridge/PLB-")
            latent2 =
            image2 = res['ui']['images']
            pixels = decoded_pil
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            path = os.path.join(folder_paths.get_temp_directory(), 'PreviewBridge', image2[0]['filename'])
            core.set_previewbridge_image(unique_id, path, image2[0])
            core.preview_bridge_image_id_map[image] = (path, image2[0])
            core.preview_bridge_image_name_map[unique_id, path] = (image, image2[0])
            core.preview_bridge_cache[unique_id] = (latent, latent2)

            image = image2

        return {
            "ui": {"images": image},
            "result": (pixels, mask, ),
        }
