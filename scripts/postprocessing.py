import sys
import os
import numpy as np
sys.path.insert(0, "src/python")
from interface import ImDataParamsRelax

filename = "data/phantom/20241202_164009_502_ImDataParamsBMRR_subspace.h5"

if os.path.exists(filename):
    obj = ImDataParamsRelax(filename)
    obj.reformat_sag2tra()
    obj.set_FatModel("phantom")
    obj.Masks["tissueMask"] = obj.get_tissueMaskFilled(threshold=0.1, iDyn=0)

    obj.run_fieldmapping(ind_dynamic=[0]) 
    obj.get_wfi_images(fieldmap_ind_dynamic=0)
    obj.save_WFIparams()

    obj.load_dictionary("src/Files/20241022_dict_caspr_lookLocker_with0deg_31B0.h5")
    obj.RelaxParams["dictionary"]["B1"] = np.array([1.0])
    obj.RelaxParams["dictionary"]["dictionary"] = obj.RelaxParams["dictionary"]["dictionary"][:,np.newaxis,...]

    obj.perform_dictionary_matching(signal=obj.WFIparams["water"], complex_signal=True, compute_pd=True, with_profiles=True)
    obj.set_relaxometry_mask(10)
    obj.save_RelaxParams()

    path = os.path.dirname(filename)    
    basename = os.path.basename(filename)[:-3]
    folder_path = f"{path}/{basename}_nifti"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    types = ["T1_ms", "T2_ms", "PD"]
    for image_type in types:
        obj.export_Array2nii(np.flip(np.transpose(np.abs(obj.RelaxParams[image_type]), [1, 0, 2]), axis=(0,1)), f"{folder_path}/{image_type}.nii")

    types = ["fat"]
    for image_type in types:
        obj.export_Array2nii(np.flip(np.transpose(np.abs(obj.WFIparams[image_type][:,:,:,0]), [1, 0, 2]), axis=(0,1)), f"{folder_path}/coeff1_{image_type}.nii")
else:
    print("File '{filename}' not found! Please reconstruct data first.")