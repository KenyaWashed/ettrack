import numpy as np
import os
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList

class DroneDataset(BaseDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.base_path = dataset_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    # def _construct_sequence(self, sequence_info):
    #     frames_dir = os.path.join(self.base_path, sequence_info['path'])
    #     frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    #     gt_path = os.path.join(self.base_path, sequence_info['anno_path'])
    #     if not os.path.exists(gt_path):
    #         raise FileNotFoundError(f"File groundtruth_rect.txt không tồn tại trong {gt_path}")
        
    #     ground_truth_rect = np.loadtxt(gt_path, delimiter=",")

    #     seq = Sequence(
    #         name=sequence_info['name'],
    #         frames=frames,
    #         dataset='drone',
    #         object_class=sequence_info.get('object_class', 'other')
    #     )
    #     seq.ground_truth_rect = ground_truth_rect
    #     return seq

    def _construct_sequence(self, sequence_info):
        frames_dir = os.path.join(self.base_path, sequence_info['path'])
        frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
        gt_path = os.path.join(self.base_path, sequence_info['anno_path'])
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"File groundtruth_rect.txt không tồn tại trong {gt_path}")
    
        ground_truth_rect = np.loadtxt(gt_path, delimiter=",")
    
        seq = Sequence(
            name=sequence_info['name'],
            frames=frames,
            dataset='drone',
            ground_truth_rect=ground_truth_rect,  # truyền thẳng ở đây
            object_class=sequence_info.get('object_class', 'other')
        )
        return seq


    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequences = []
        for vid in os.listdir(self.base_path):
            video_dir = os.path.join(self.base_path, vid)
            frames_dir = os.path.join(video_dir, "frames")
            gt_file = os.path.join(frames_dir, "groundtruth_rect.txt")
            if os.path.exists(frames_dir) and os.path.exists(gt_file):
                sequences.append({
                    'name': vid,
                    'path': os.path.join(vid, "frames"),
                    'anno_path': os.path.join(vid, "frames", "groundtruth_rect.txt"),
                    'object_class': 'other'
                })
        return sequences
