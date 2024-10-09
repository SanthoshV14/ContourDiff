import SimpleITK as sitk
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import numpy as np
from glob import glob
import argparse
import cv2

def get_data_array(modality, thresh, img_size=512):
    data_array = None
    hu_min, hu_max = -1.024, 1.024
    
    def get_mod(hu):
        hu = apply_modality_lut(hu.pixel_array, hu)/1000.
        hu = np.clip(hu, a_min=hu_min, a_max=hu_max)
        hu = ((hu - hu_min)/(hu_max - hu_min)) * 255.
        hu = np.array(hu, dtype=np.uint8)
        yield hu

    n_patients = len(glob(f'../data/dicoms/*/'))
    for pelvic_index in range(1, n_patients+1):
        array = None
        path = f'../data/dicoms/Pelvic-Ref-{pelvic_index:03d}/{modality}/*'
        n_files = len(glob(path))
        if n_files >= thresh:
            mid = n_files//2
            files = sorted(glob(path))[mid-thresh//2: mid+thresh//2]
            for i, file in enumerate(files):
                hu = pydicom.dcmread(file)
                hu = next(get_mod(hu))
                hu = cv2.resize(hu, (img_size, img_size))
                hu = np.expand_dims(hu, axis=0)
                hu = np.expand_dims(hu, axis=0)
                array = hu if array is None else np.concatenate((array, hu), axis=1)
            data_array = array if data_array is None else np.concatenate((data_array, array), axis=0)
    
    return data_array

def get_masks_array(modality, thresh, img_size=512):
    masks_array = None
    n_patients = len(glob(f'../data/dicoms/*/'))
    for pelvic_index in range(1, n_patients+1):
        path = f'../data/masks/Pelvic-Ref-{pelvic_index:03d}/converted_{modality}/skin.mha'
        masks = sitk.ReadImage(path, imageIO="MetaImageIO")
        masks = sitk.GetArrayFromImage(masks)[::-1, :, :]
        if masks.shape[0] >= thresh:
            mid = masks.shape[0]//2
            masks = masks[mid-thresh//2: mid+thresh//2, :, :]
            masks = cv2.resize(masks.transpose(1, 2, 0), (img_size, img_size))
            masks = masks.transpose(2, 0, 1)
            masks = np.expand_dims(masks, axis=0)
            masks_array = masks if masks_array is None else np.concatenate((masks_array, masks), axis=0)
    
    return masks_array

def get_edges(data):

    def canny_edge_detector(image, low_threshold=30, high_threshold=50, kernel_size=3):
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        magnitude = cv2.Canny(blurred_image, low_threshold, high_threshold)
        yield magnitude

    edges = None
    for patient in data:
        patient_edge = None
        for slice in patient:
            edge = next(canny_edge_detector(slice))
            edge = np.expand_dims(edge, axis=0)
            edge = np.expand_dims(edge, axis=0)
            patient_edge = edge if patient_edge is None else np.concatenate((patient_edge, edge), axis=1)

        edges = patient_edge if edges is None else np.concatenate((edges, patient_edge), axis=0)
    
    return edges

def save_npz(output_dir: str, ct, ct_masks, cbct, cbct_masks):
    ct = ct * ct_masks
    cbct = cbct * cbct_masks
    np.savez(file=output_dir, ct=ct, ct_masks=ct_masks, cbct=cbct, cbct_masks=cbct_masks)

def main(agrs):
    ct = get_data_array(args.ct_dir, args.ct_thresh, args.img_size)
    cbct = get_data_array(args.cbct_dir, args.cbct_thresh, args.img_size)
    ct_masks = get_masks_array(args.ct_dir, args.ct_thresh, args.img_size)
    cbct_masks = get_masks_array(args.cbct_dir, args.cbct_thresh, args.img_size)
    print(ct.shape, ct_masks.shape, cbct.shape, cbct_masks.shape)
    save_npz(f'{agrs.output_dir}_{args.img_size}', ct, ct_masks, cbct, cbct_masks)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ct_dir', type=str, default=None, help="CT directory name")
    parser.add_argument('--cbct_dir', type=str, default=None, help="CBCT directory name")
    parser.add_argument('--ct_thresh', type=str, default=120, help="No of CT slices per patient")
    parser.add_argument('--cbct_thresh', type=str, default=80, help="No of CBCT slices per patient")
    parser.add_argument('--output_dir', type=str, default="../data", help="output file name")
    parser.add_argument('--img_size', type=int, default=512, help="output image size default=512")
    args = parser.parse_args()

    main(args)

