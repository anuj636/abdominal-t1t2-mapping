import sys
import os
import numpy as np
sys.path.insert(0, "src/python")
from interface import ImDataParamsRelax

filename = "data/phantom/20260126_094752_csBFFE_RLT_HR_ImDataParamsBMRR_subspace.h5"

if os.path.exists(filename):
    obj = ImDataParamsRelax(filename)
    obj.reformat_sag2tra()
    obj.set_FatModel("phantom")
    mask = obj.get_tissueMaskFilled(threshold=0.1, iDyn=0)
    # hmrGC's trim_zeros walks inward from each edge until it finds a non-zero plane.
    # Guarantee a zero border only on dimensions where interior voxels exist.
    if mask.shape[0] > 1:
        mask[0, :, :] = False
        mask[-1, :, :] = False
    if mask.shape[1] > 1:
        mask[:, 0, :] = False
        mask[:, -1, :] = False
    if mask.shape[2] > 1:
        mask[:, :, 0] = False
        mask[:, :, -1] = False
    if not np.any(mask):
        sig = obj.ImDataParams["signal"][:, :, :, 0, :]
        raise RuntimeError(
            "tissueMask is empty for dynamic 0. "
            f"Signal max magnitude is {float(np.max(np.abs(sig))):.6g}. "
            "Field-mapping cannot run on empty/zero signal."
        )
    obj.Masks["tissueMask"] = mask

    obj.run_fieldmapping(ind_dynamic=[0]) 
    obj.get_wfi_images(fieldmap_ind_dynamic=0)
    obj.save_WFIparams(mat_file=False)

    obj.load_dictionary("src/Files/20241022_dict_caspr_lookLocker_with0deg_31B0.h5")
    obj.RelaxParams["dictionary"]["B1"] = np.array([1.0])
    obj.RelaxParams["dictionary"]["dictionary"] = obj.RelaxParams["dictionary"]["dictionary"][:,np.newaxis,...]

    use_profiles = "subspaceBasis" in obj.ImDataParams
    if not use_profiles:
        print("subspaceBasis not found in reconstruction output; falling back to dictionary matching without profiles.")
    else:
        n_signal_dyn = obj.ImDataParams["signal"].shape[3]
        n_subspace = obj.ImDataParams["subspaceBasis"].shape[0]
        if n_signal_dyn != n_subspace:
            import warnings
            warnings.warn(
                f"Reconstruction output has {n_signal_dyn} dynamics but subspaceBasis has "
                f"{n_subspace} components. The reconstruction should be re-run with exactly "
                f"{n_subspace} subspace components to match subspaceBasis. "
                f"Dictionary matching will proceed with signal truncated to {n_subspace} "
                f"features, which is valid only if both bases share the same SVD ordering.",
                stacklevel=1
            )

    obj.perform_dictionary_matching(
        signal=obj.WFIparams["water"],
        complex_signal=True,
        compute_pd=True,
        with_profiles=use_profiles,
    )
    obj.set_relaxometry_mask(10)
    obj.save_RelaxParams(mat_file=False)

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