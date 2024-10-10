from torch.utils.data import Dataset
import torch
import numpy as np
import cv2

class CTCBCTDataset(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.data = np.load(path)
        self.ct = self.data['ct']
        # self.cbct = self.data['cbct']
        self.ct_masks = self.data['ct_masks']
        # self.cbct_masks = self.data['cbct_masks']
        self.n_patients = self.ct.shape[0]
        self.n_slices = self.ct.shape[1]
        self.length = self.n_patients * self.n_slices
    
    def canny_edge_detector(self, image, low_threshold=30, high_threshold=50, kernel_size=3):
        image = image.numpy().astype(np.uint8)
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        magnitude = cv2.Canny(blurred_image, low_threshold, high_threshold)
        return torch.tensor(magnitude)

    def __len__(self):
        return self.n_patients * self.n_slices
    
    def __getitem__(self, index):
        index = index if index >= 0 else self.length + index
        nth_patient = index//(self.n_slices)
        nth_slice = index%(self.n_slices)
        ct = torch.tensor(self.ct[nth_patient, nth_slice, :, :]).unsqueeze(0).to(torch.float32)/255.
        ct_masks = torch.tensor(self.ct_masks[nth_patient, nth_slice, :, :]).unsqueeze(0).to(torch.float32)
        ct_contours = self.canny_edge_detector(ct.permute(1, 2, 0)).unsqueeze(0).to(torch.float32)
        
        return {
            "ct": ct,
            "ct_masks": ct_masks,
            "ct_contours": ct_contours,
            "nth_patient": torch.tensor([nth_patient], dtype=torch.int32),
            "nth_slice": torch.tensor([nth_slice], dtype=torch.int32)
        }