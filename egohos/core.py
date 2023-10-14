import os
import sys
import glob
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import cv2
from PIL import Image, ImageOps

#detic_path = os.getenv('DETIC_PATH') or 'Detic'
#sys.path.insert(0,  detic_path)

from mmseg.apis import init_segmentor
from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose
# from .checkpoint import ensure_checkpoint
from torchvision.ops import masks_to_boxes

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_DIR = os.path.join(os.getenv("MODEL_DIR") or 'models', 'egohos')

DUMB_META_KEYS = {k: '' for k in ['additional_channel', 'twohands_dir', 'cb_dir']}
DUMB_ARG_KEYS = {'filename': '__.png', 'ori_filename': '__.png'}

COLORS = np.array([
    (0,    0,   0),     # background
    (255,  0,   0),     # left_hand
    (0,    0,   255),   # right_hand
    (255,  0,   255),   # left_object1
    (0,    255, 255),   # right_object1
    (0,    255, 0),     # two_object1
    (255,  204, 255),   # left_object2
    (204,  255, 255),   # right_object2
    (204,  255, 204),   # two_object2
])

def desc(x):
    if isinstance(x, dict):
        return {k: desc(v) for k, v in x.items()}
    if isinstance(x, (dict, list, tuple, set)):
        return type(x)(desc(xi) for xi in x)
    if hasattr(x, 'shape'):
        return f'{type(x).__name__}({x.shape}, {x.dtype})'
    return x

class BaseEgoHos(nn.Module):
    CLASSES = np.array(['', 'hand(left)', 'hand(right)', 'obj1(left)', 'obj1(right)', 'obj1(both)', 'obj2(left)', 'obj2(right)', 'obj2(both)', 'cb'])
    CLASS_MAP = np.array([])

    def __init__(self, config=None, checkpoint=None, device=device):
        super().__init__()
        assert config is not None, "You must provide a config"
        if not os.path.isfile(config):
            self.name = config
            config = os.path.join(MODEL_DIR, config, f'{config}.py')
            assert os.path.isfile(config), f'{self.name} not in {os.listdir(MODEL_DIR)}'
        else:
            self.name = os.path.basename(os.path.dirname(config))
        if not checkpoint:
            checkpoint = max(glob.glob(os.path.join(os.path.dirname(config), '*.pth')))
        
        #print('using device:', device)
        if device == 'cpu':
            import mmcv
            config = mmcv.Config.fromfile(config)
            config["norm_cfg"]["type"] = "BN"
            config["model"]["backbone"]["norm_cfg"]["type"] = "BN"
            config["model"]["decode_head"]["norm_cfg"]["type"] = "BN"
            config["model"]["auxiliary_head"]["norm_cfg"]["type"] = "BN"
        self.model = init_segmentor(config, checkpoint, device=device)
        self.preprocess = Compose(self.model.cfg.data.test.pipeline[1:])

        self.device = device
        self.is_cuda = device != 'cpu'
        # self.palette = get_palette(None, self.model.CLASSES)
        # self.classes = self.model.CLASSES
        self.addt_model=None

        self.CLASS_IDS = np.arange(1, len(self.CLASS_MAP))

    def forward(self, img, addt=None, include_addt=True, internal=False):
        x_img, img_meta = self._prepare_input(img)
        
        # add additional segmentation maps
        if addt is None and self.addt_model is not None:
            addt = self.addt_model(img, internal=True)
        if addt is not None:
            x_addt = torch.flip(addt, [1])
            x_addt = self.pad_resize(x_addt, x_img)
            x_img = torch.cat([x_img, x_addt], dim=1)

        # run model
        seg_logit = self.model.inference(x_img, img_meta, rescale=True)
        result = seg_logit.argmax(dim=1, keepdim=True)
        
        if internal:
            # merge with input seg masks
            if include_addt and addt is not None:
                result = torch.cat([result, addt], dim=1)
            return result
        return self.as_instance_masks(result, addt)

    def as_instance_masks(self, id_mask, addt): # id_mask: [1, 1, H, W], addt: [1, c, H, W]
        mask, cids = as_instance_masks(id_mask[0, 0], self.CLASS_IDS)
        cids = np.atleast_1d(self.CLASS_MAP[cids]) # force 1d

        if addt is not None:
            mask2, cids2 = self.addt_model.as_instance_masks(addt[:, :1], addt[:, 1:] if addt.shape[1]>1 else None)
            mask = torch.cat([mask, mask2])
            cids = np.concatenate([cids, cids2])
        return mask, cids

    def _prepare_input(self, img):
        data = {'img': img, 'img_shape': img.shape, 'ori_shape': img.shape, **DUMB_ARG_KEYS}
        data = self.preprocess(data)
        img = data['img'][0].to(self.device)[None]
        img_meta = [data['img_metas'][0].data]
        img_meta[0].update(DUMB_META_KEYS)
        # data['img_metas'] = [[i.data] for i in data['img_metas']]
        # data['img_metas'][0][0].update(DUMB_META_KEYS)
        return img, img_meta

    def pad_resize(self, aux, im):
        _, _, h, w = im.shape
        _, _, ha, wa = aux.shape
        hn, wn = (int(h/w*wa), wa) if ha/wa < h/w else (ha, int(ha/(h/w)))
        dh, dw = (hn-ha)/2, (wn-wa)/2
        
        ps = (int(dh), int(dw), int(np.ceil(dh)), int(np.ceil(dw)))
        aux = F.pad(aux, ps, "constant", value=0)
        aux = F.interpolate(aux.float(), (h, w))
        return torch.Tensor(aux)

def as_instance_masks(id_mask, class_ids): # result: [H, W]
    mask = torch.zeros(
        (len(class_ids), *id_mask.shape), 
        dtype=bool, device=id_mask.device)
    for i, c in enumerate(class_ids):
        mask[i] = id_mask == c
    valid = mask.any(1).any(1)
    mask = mask[valid]
    class_ids = np.atleast_1d(class_ids[valid.cpu().bool().numpy()]) # force 1d, also doesnt work without explicit bool cast
    return mask, class_ids


class EgoHosHands(BaseEgoHos):
    # output_names = ('hands',)
    CLASS_MAP = np.array([0, 1, 2])
    def __init__(self, config='seg_twohands_ccda', **kw):
        super().__init__(config, **kw)

class EgoHosCB(BaseEgoHos):
    # output_names = ('cb', 'hands')
    CLASS_MAP = np.array([0, 9])
    def __init__(self, config='twohands_to_cb_ccda', addt=True, **kw):
        super().__init__(config, **kw)
        self.addt_model = EgoHosHands() if addt else None

    # def as_instance_masks(self, result, addt):
    #     mask = result.bool()[None]
    #     class_ids = np.array([9])
    #     valid = int(mask.any())
    #     return mask[:valid], class_ids[:valid]

class EgoHosObj1(BaseEgoHos):
    # output_names = ('obj1', 'cb', 'hands')
    CLASS_MAP = np.array([0, 3, 4, 5])
    def __init__(self, config='twohands_cb_to_obj1_ccda', addt=True, **kw):
        super().__init__(config, **kw)
        self.addt_model = EgoHosCB() if addt else None

class EgoHosObj2(BaseEgoHos):
    # output_names = ('obj2', 'cb', 'hands')
    CLASS_MAP = np.array([0, 3, 4, 5, 6, 7, 8])
    def __init__(self, config='twohands_cb_to_obj2_ccda', addt=True, **kw):
        super().__init__(config, **kw)
        self.addt_model = EgoHosCB() if addt else None

# class EgoHosObjs(nn.Module):
#     output_names = ('obj1', 'obj2', 'cb', 'hands')
#     def __init__(self):
#         super().__init__()
#         self.addt_model = EgoHosCB()
#         self.obj1_model = EgoHosObj1(addt=False)
#         self.obj2_model = EgoHosObj2(addt=False)
#         self.classes = self.obj2_model.classes

#     def forward(self, im, addt=None, internal=False):
#         if addt is None:
#             addt = self.addt_model(im, internal=True)
#         obj1 = self.obj1_model(im, addt, include_addt=False, internal=True)
#         obj2 = self.obj2_model(im, addt, include_addt=False, internal=True)
#         if internal:
#             return [np.concatenate(xs) for xs in zip(obj1, obj2, addt)]

#         obj1, obj1_class_ids = self.obj1_model.as_instance_masks(obj1)
#         obj2, obj2_class_ids = self.obj2_model.as_instance_masks(obj1)
        


MODELS = {'hands': EgoHosHands, 'obj1': EgoHosObj1, 'obj2': EgoHosObj2, 'cb': EgoHosCB}

class EgoHos(BaseEgoHos):
    def __new__(cls, mode='obj2', *a, **kw):
        return MODELS[mode](*a, **kw)
  

def merge_segs(segs):
    out = np.zeros_like(segs[0])
    for s in segs:
        where = s != 0
        out[where] = s[where]
    return out


# def get_palette(palette, classes):
#     if palette is None:
#         state = np.random.get_state()
#         np.random.seed(42)
#         palette = np.random.randint(0, 255, size=(len(classes), 3))
#         np.random.set_state(state)
#     palette = np.array(palette)
#     assert palette.shape == (len(classes), 3)
#     return palette[:,::-1]


# def draw_segs(im, result, palette, opacity=0.5):
#     seg = result[0]
#     assert 0 < opacity <= 1.0
#     opacity = (seg[...,None]!=0)*opacity
#     im = im * (1 - opacity) + palette[seg] * opacity
#     return im.astype(np.uint8)


#from IPython import embed
@torch.no_grad()
def run(src, size=480, out_file=True, **kw):
    """Run multi-target tracker on a particular sequence.
    """
    import tqdm
    import supervision as sv

    model = EgoHos(**kw)
    print(model.CLASSES)
    # print(model.model.cfg['additional_channel'])
    classes = np.array(list(model.CLASSES))
    class_ids = np.arange(len(classes))

    if out_file is True:
        out_file = f'egohos{model.name}_{os.path.basename(src)}'

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    # video_info = sv.VideoInfo.from_video_path(src)
    video_info, WH = get_video_info(src, size=size)
    with sv.VideoSink(out_file, video_info=video_info) as s:
        for i, im in tqdm.tqdm(enumerate(sv.get_video_frames_generator(src)), total=video_info.total_frames):
            im = cv2.resize(im, WH)
            masks, class_ids = model(im)
            boxes = masks_to_boxes(torch.as_tensor(masks))
            detections = sv.Detections(
                xyxy=boxes.cpu().numpy(),
                mask=masks.cpu().numpy(),
                class_id=class_ids,
            )
            im = mask_annotator.annotate(
                scene=im,
                detections=detections,
            )
            im = box_annotator.annotate(
                scene=im,
                detections=detections,
                labels=[
                    f"{classes[class_id]}"
                    for _, _, _, class_id, _
                    in detections
                ]
            )
            s.write_frame(im)


def get_video_info(src, size=None, fps_down=1, nrows=1, ncols=1):
    import supervision as sv
    # get the source video info
    video_info = sv.VideoInfo.from_video_path(video_path=src)
    # make the video size a multiple of 16 (because otherwise it won't generate masks of the right size)
    aspect = video_info.width / video_info.height
    size = size or video_info.height
    video_info.width = int(aspect*size)//16*16
    video_info.height = int(size)//16*16
    WH = video_info.width, video_info.height

    # double width because we have both detic and xmem frames
    video_info.width *= ncols
    video_info.height *= nrows
    # possibly reduce the video frame rate
    video_info.og_fps = video_info.fps
    video_info.fps /= fps_down or 1

    print(f"Input Video {src}\nsize: {WH}  fps: {video_info.fps}")
    return video_info, WH



if __name__ == '__main__':
    import fire
    fire.Fire(run)

