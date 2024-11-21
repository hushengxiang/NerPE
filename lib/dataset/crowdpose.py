# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os

from crowdposetools.coco import COCO
from crowdposetools.cocoeval import COCOeval
import json_tricks as json
import numpy as np

from lib.dataset.JointsDataset import JointsDataset
from lib.nms.nms import oks_nms, soft_oks_nms


logger = logging.getLogger(__name__)


class CrowdPoseDataset(JointsDataset):
    '''
    "keypoints": {
        0: 'left_shoulder',
        1: 'right_shoulder',
        2: 'left_elbow',
        3: 'right_elbow',
        4: 'left_wrist',
        5: 'right_wrist',
        6: 'left_hip',
        7: 'right_hip',
        8: 'left_knee',
        9: 'right_knee',
        10: 'left_ankle',
        11: 'right_ankle',
        12: 'top_head',
        13: 'neck'
    }
    '''
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        extra = cfg.TEST.EXTRA
        self.nms_thre = extra.NMS_THRE
        self.image_thre = extra.IMAGE_THRE
        self.soft_nms = extra.SOFT_NMS
        self.oks_thre = extra.OKS_THRE
        self.in_vis_thre = extra.IN_VIS_THRE
        self.use_gt_bbox = extra.USE_GT_BBOX
        self.bbox_file = extra.COCO_BBOX_FILE

        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self.num_joints = 14
        self.flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 12, 13)
        self.lower_body_ids = (6, 7, 8, 9, 10, 11)

        self.sigmas = np.array([
            .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79
        ]) / 10.0

        self.coco = COCO(self._get_ann_file_keypoint())

        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [(self._class_to_coco_ind[cls], self._class_to_ind[cls]) for cls in self.classes[1:]]
        )

        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)
        logger.info('=> num_images: {}'.format(self.num_images))
        
        self.db = self._get_db()
        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_ann_file_keypoint(self):
        """ self.root / annotations / crowdpose_{train,val,test}.json """
        return os.path.join(self.root, 'annotations', 'crowdpose_{}.json'.format(self.image_set))

    def _get_db(self):
        if self.is_train or self.use_gt_bbox:
            gt_db = self._load_coco_keypoint_annotations()
        else:
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for img_id in self.img_ids:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(img_id))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, img_id):
        im_ann = self.coco.loadImgs(img_id)[0]
        width = im_ann['width']
        height = im_ann['height']

        # set iscrowd to None or False
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float32)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._xywh2cs(*obj['clean_bbox'][:4])
            rec.append({
                'image': self.image_path_from_id(img_id),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        return rec

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_id(self, img_id):
        """ example: images / 100000.jpg """
        file_name = '%d.jpg' % (img_id)
        image_path = os.path.join(self.root, 'images', file_name)

        return image_path

    def _load_coco_person_detection_results(self):
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        logger.info('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = self.image_path_from_id(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._xywh2cs(*box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float32)
            kpt_db.append({
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        logger.info('=> Total boxes after fliter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make {}'.format(res_folder))

        res_file = os.path.join(
            res_folder, 'keypoints_{}_results.json'.format(self.image_set)
        )

        # person x keypoints
        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image_id': int(img_path[idx][-10:-4])
            })
        # image x person x keypoints
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image_id']].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.soft_nms:
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre, sigmas=self.sigmas
                )
            else:
                keep = oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre, sigmas=self.sigmas
                )

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(oks_nmsed_kpts, res_file)

        info_str = self._do_python_keypoint_eval(res_file)
        name_value = OrderedDict(info_str)

        return name_value, name_value['AP']

    def _write_coco_keypoint_results(self, keypoints, res_file):
        """ Write results into a json file """
        data_pack = [{
            'cat_id': self._class_to_coco_ind[cls],
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        } for cls_ind, cls in enumerate(self.classes) if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        logger.info('=> writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """ Get coco keypoint results """
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))])
            key_points = _key_points.reshape(-1, self.num_joints * 3)

            result = [{
                'image_id': img_kpts[k]['image_id'],
                'category_id': cat_id,
                'keypoints': list(key_points[k]),
                'score': img_kpts[k]['score'],
                'center': list(img_kpts[k]['center']),
                'scale': list(img_kpts[k]['scale'])
            } for k in range(len(img_kpts))]

            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """ Keypoint evaluation using COCOAPI """
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_det, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'AP .5', 'AP .75', 'AR', 'AR .5', 'AR .75', 'AP(E)', 'AP(M)', 'AP(H)']
        stats_index = [0, 1, 2, 5, 6, 7, 8, 9, 10]

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[stats_index[ind]]))

        return info_str