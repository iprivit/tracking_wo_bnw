from collections import OrderedDict

import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes

import numpy as np


class FRCNN_FPN: # (FasterRCNN):
    def __init__(self, model, num_classes=1):
#     def __init__(self, num_classes):
#         backbone = resnet_fpn_backbone('resnet50', False)
#         backbone = model.model.backbone
#         super(FRCNN_FPN, self).__init__(backbone, num_classes)
        self.backbone = model.model.backbone
        self.model = model    
        # these values are cached to allow for feature reuse
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None

    def detect(self, img):
#         device = list(self.parameters())[0].device
#         img = img.to(device)
#         detections = self(img)[0]
#         return detections['boxes'].detach(), detections['scores'].detach()
        device = self.model.cfg['MODEL']['DEVICE']

        try:
            detections = self.model(img)
        except:
            shape = img.shape
#             print('img shape: ',img.shape)
            reshaped_img = np.reshape(img.cpu().numpy().squeeze(),(shape[2],shape[3],shape[1]))
            reshaped_img = np.array(reshaped_img, dtype=np.uint8)
#             print('reshaped img: ',reshaped_img.shape)
            detections = self.model(reshaped_img) # FIX THIS 

        return detections["instances"].get_fields()['pred_boxes'].tensor.detach(), detections["instances"].get_fields()['scores'].detach()

    def predict_boxes(self, boxes):
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)

        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_images.image_sizes[0])
        proposals = [boxes]

        box_features = self.roi_heads.box_roi_pool(self.features, proposals, self.preprocessed_images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)

        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        # score_thresh = self.roi_heads.score_thresh
        # nms_thresh = self.roi_heads.nms_thresh

        # self.roi_heads.score_thresh = self.roi_heads.nms_thresh = 1.0
        # self.roi_heads.score_thresh = 0.0
        # self.roi_heads.nms_thresh = 1.0
        # detections, detector_losses = self.roi_heads(
        #     features, [boxes.squeeze(dim=0)], images.image_sizes, targets)

        # self.roi_heads.score_thresh = score_thresh
        # self.roi_heads.nms_thresh = nms_thresh

        # detections = self.transform.postprocess(
        #     detections, images.image_sizes, original_image_sizes)

        # detections = detections[0]
        # return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()

        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        return pred_boxes, pred_scores

    def load_image(self, images):
        device = self.model.cfg['MODEL']['DEVICE']
#         device = list(self.parameters())[0].device    
        try:
            images = images.to(device)
        except:
            images = torch.tensor(images,device=device)

        self.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images = self.normalize(images)  # FIX THIS
        preprocessed_images = self.resize(preprocessed_images)
        self.preprocessed_images = preprocessed_images

#         self.original_image_sizes = [img.shape[-2:] for img in images]
#         preprocessed_images, _ = self.transform(images, None)
#         self.preprocessed_images = preprocessed_images

#         self.features = self.backbone(preprocessed_images.tensors)
        self.features = self.backbone(torch.unsqueeze(preprocessed_images,0)) #.tensors

        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])
            
    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.model.cfg['MODEL']['PIXEL_MEAN'], dtype=dtype, device=device)
        std = torch.as_tensor(self.model.cfg['MODEL']['PIXEL_STD'], dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]
    
    def resize(self, image, target=None):
        # type: (Tensor, Optional[Dict[str, Tensor]])
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = float(torch.min(im_shape))
        max_size = float(torch.max(im_shape))
#         if self.training:  ### COMMENTING OUT FOR NOW
#             size = float(self.torch_choice(self.min_size))
#         else:
            # FIXME assume for now that testing uses the largest scale
        size = float(self.model.cfg['INPUT']['MIN_SIZE_TEST'])
        scale_factor = size / min_size
        if max_size * scale_factor > self.model.cfg['INPUT']['MAX_SIZE_TEST']:
            scale_factor = self.model.cfg['INPUT']['MAX_SIZE_TEST'] / max_size
        image = torch.nn.functional.interpolate(
            image, size = 1024, mode='bilinear', align_corners=False)[0] # had to remove the [None] part
        # removed scale_factor=scale_factor

        if target is None:
            return image #, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        if "masks" in target:
            mask = target["masks"]
            mask = misc_nn_ops.interpolate(mask[None].float(), scale_factor=scale_factor)[0].byte()
            target["masks"] = mask

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints
        return image #, target
    
            
class box_class:
    def __init__(self, tensor):
        self.tensor = tensor
    
    def area(self):
        area = []
        for box in self.tensor:
            dif1 = box[2]-box[0]
            dif2 = box[3]-box[1]
            ar = dif1*dif2
            area.append(ar)
        area = np.sum(area)   
        return area
    
# [[ 340.,  347.,  366.,  387.],
#          [ 193.,  385.,  223.,  413.],
#          [  77.,  333.,  106.,  369.],
#          [ 472.,  345.,  498.,  385.],
#          [ 594.,  182.,  620.,  219.],
#          [1091.,  356., 1123.,  384.],
#          [ 897.,  335.,  923.,  371.],
#          [ 608.,  378.,  635.,  414.],
#          [ 748.,  343.,  773.,  382.],
#          [ 680.,  465.,  708.,  492.],
#          [ 507.,  420.,  531.,  435.],
#          [ 327.,  460.,  357.,  486.],
#          [1266.,  625., 1297.,  656.],
#          [ 777.,  214.,  805.,  248.]]
            
class MRCNN_FPN:

    def __init__(self, model, num_classes=1):
        self.backbone = model.model.backbone
        self.model = model
        #super(MRCNN_FPN, self).__init__(backbone,num_classes)
        # these values are cached to allow for feature reuse
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None

    def detect(self, img):
        device = self.model.cfg['MODEL']['DEVICE']
        #img = img.to(device)

        try:
            detections = self.model(img)
        except:
            shape = img.shape
            reshaped_img = np.reshape(img.cpu().numpy().squeeze(),(shape[2],shape[3],shape[1]))
            reshaped_img = np.array(reshaped_img, dtype=np.uint8)
            detections = self.model(reshaped_img) # FIX THIS 

        return detections["instances"].get_fields()['pred_boxes'].tensor.detach(), detections["instances"].get_fields()['scores'].detach()

    def predict_boxes(self, boxes):
        device = self.model.cfg['MODEL']['DEVICE']
        boxes = boxes.to(device)

        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_images.shape)
        proposals = [box_class(box) for box in boxes]
        boxes = box_class(boxes)
        proposals = [boxes] # proposals lookg ood 
        
#         print('proposals: ',proposals[0].tensor)
#         print('feature keys: ',self.features.keys())
#         print('features: ',self.features)
#         print('image_shape: ',self.preprocessed_images.shape)
        try:
            box_features = self.model.model.roi_heads.box_pooler(self.features, proposals) # image_sizes  # self.preprocessed_images.shape
        except:
#             print('trying pure list')
            feat_list = [self.features[k] for k in self.features][:-1]
#             for box in boxes: 
#                 boxc = box_class(box)
            box_features = self.model.model.roi_heads.box_pooler(feat_list, proposals) # image_sizes  # self.preprocessed_images.shape
        #box_features = self.roi_heads.box_roi_pool(self.features, proposals, self.preprocessed_images.image_sizes)

        #self.features, # removed self.features
        box_features = self.model.model.roi_heads.box_head(box_features)
        class_logits, box_regression = self.model.model.roi_heads.box_predictor(box_features)
#         print(box_regression)

#         pred_boxes = self.model.model.roi_heads.box_coder.decode(box_regression, proposals) # failing here 
        pred_boxes = proposals[0].tensor #.detach()
        pred_scores = F.softmax(class_logits, -1)

#         pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.shape, self.original_image_sizes[0]) # image_sizes[0]
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        return pred_boxes, pred_scores
    
    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.model.cfg['MODEL']['PIXEL_MEAN'], dtype=dtype, device=device)
        std = torch.as_tensor(self.model.cfg['MODEL']['PIXEL_STD'], dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]
    
    def resize(self, image, target=None):
        # type: (Tensor, Optional[Dict[str, Tensor]])
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = float(torch.min(im_shape))
        max_size = float(torch.max(im_shape))
#         if self.training:  ### COMMENTING OUT FOR NOW
#             size = float(self.torch_choice(self.min_size))
#         else:
            # FIXME assume for now that testing uses the largest scale
        size = float(self.model.cfg['INPUT']['MIN_SIZE_TEST'])
        scale_factor = size / min_size
        if max_size * scale_factor > self.model.cfg['INPUT']['MAX_SIZE_TEST']:
            scale_factor = self.model.cfg['INPUT']['MAX_SIZE_TEST'] / max_size
        image = torch.nn.functional.interpolate(
            image, size = 1024, mode='bilinear', align_corners=False)[0] # had to remove the [None] part
        # removed scale_factor=scale_factor

        if target is None:
            return image #, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        if "masks" in target:
            mask = target["masks"]
            mask = misc_nn_ops.interpolate(mask[None].float(), scale_factor=scale_factor)[0].byte()
            target["masks"] = mask

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints
        return image #, target

    def load_image(self, images):
        device = self.model.cfg['MODEL']['DEVICE']
        try:
            images = images.to(device)
        except:
            images = torch.tensor(images,device=device)

        self.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images = self.normalize(images)  # FIX THIS
        preprocessed_images = self.resize(preprocessed_images)
        self.preprocessed_images = preprocessed_images

        self.features = self.backbone(torch.unsqueeze(preprocessed_images,0)) #.tensors
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])
