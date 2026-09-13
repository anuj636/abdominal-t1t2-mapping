import sys
import os
import numpy as np
from pathlib import Path
sys.path.insert(0, "src/python")
from interface import ImDataParamsRelax
from dictionary_matching import prepare_dictionary


def restore_collapsed_dynamics(obj, collapsed_block=28):
    signal = obj.ImDataParams.get("signal")
    if signal is None or signal.ndim != 5:
        return

    num_dyn = signal.shape[3]
    if num_dyn <= collapsed_block or num_dyn % collapsed_block != 0:
        return

    restored_dyn = num_dyn // collapsed_block
    spatial_shape = signal.shape[:3]
    num_echoes = signal.shape[4]
    reshaped = signal.reshape(*spatial_shape, restored_dyn, collapsed_block, num_echoes)
    obj.ImDataParams["signal"] = np.mean(reshaped, axis=4, dtype=signal.dtype)
    print(
        f"Collapsed dynamics detected: reducing {num_dyn} states into {restored_dyn} averaged dynamics "
        f"using contiguous blocks of {collapsed_block}."
    )


def should_collapse_dynamics() -> bool:
    return os.environ.get("POSTPROC_COLLAPSE_DYNAMICS", "0") == "1"


def should_reduce_signal_features() -> bool:
    return os.environ.get("POSTPROC_ENABLE_FEATURE_REDUCTION", "1") == "1"


def should_align_reference_grid() -> bool:
    return os.environ.get("POSTPROC_ALIGN_REFERENCE_GRID", "1") == "1"


def resize_nn_3d(arr: np.ndarray, target_shape):
    src = np.asarray(arr)
    tz, ty, tx = [int(v) for v in target_shape]
    sz, sy, sx = src.shape
    iz = np.minimum((np.arange(tz) * sz / tz).astype(int), sz - 1)
    iy = np.minimum((np.arange(ty) * sy / ty).astype(int), sy - 1)
    ix = np.minimum((np.arange(tx) * sx / tx).astype(int), sx - 1)
    return src[np.ix_(iz, iy, ix)]


def homomorphic_bias_correct(arr, mask, sigma=8):
    """Strip slow-varying coil-sensitivity/receive shading from a magnitude map.

    Uses normalized-convolution (Gaussian-weighted) smoothing restricted to the
    tissue mask to estimate a low-frequency bias field, then divides it out.
    See experiment/TODO.md "Track C" (2026-09-13) for offline validation showing
    this more than doubles PD NCC vs reference (0.1111 -> 0.2595 at sigma=8).
    """
    from scipy.ndimage import gaussian_filter

    arr = np.abs(np.asarray(arr))
    valid = mask & (arr > 0)
    num = gaussian_filter(np.where(valid, arr, 0.0), sigma=sigma)
    den = gaussian_filter(valid.astype(np.float64), sigma=sigma)
    bias = num / np.maximum(den, 1e-9)
    corrected = np.zeros_like(arr)
    corrected[valid] = arr[valid] / np.maximum(bias[valid], 1e-9)
    return corrected


def reduce_signal_features_to_target(signal, target_features):
    if signal.ndim < 1:
        raise RuntimeError("Signal array has invalid shape for feature reduction.")

    n_signal_features = int(signal.shape[-1])
    if n_signal_features == target_features:
        print(
            f"Feature contract already aligned: signal={n_signal_features}, dictionary={target_features}."
        )
        return signal

    if target_features <= 0:
        raise RuntimeError(f"Invalid target feature count: {target_features}")

    if n_signal_features % target_features != 0:
        raise RuntimeError(
            "Cannot align feature contract with deterministic block reduction: "
            f"signal features={n_signal_features}, dictionary features={target_features}."
        )

    reduce_factor = n_signal_features // target_features
    reduced = signal.reshape(*signal.shape[:-1], target_features, reduce_factor).mean(axis=-1)
    print(
        "Feature reduction applied: "
        f"signal features {n_signal_features} -> {target_features} "
        f"using contiguous block mean with factor {reduce_factor}."
    )
    return reduced


def project_signal_to_subspace(signal, subspace_basis):
    signal = np.asarray(signal)
    basis = np.asarray(subspace_basis)

    if signal.ndim < 1:
        raise RuntimeError("Signal array has invalid shape for subspace projection.")
    if basis.ndim != 2:
        raise RuntimeError(
            f"subspaceBasis must be 2D for projection, got shape {basis.shape}."
        )

    n_signal_features = int(signal.shape[-1])

    # Accept either [features, components] or [components, features].
    if basis.shape[0] == n_signal_features:
        basis_fc = basis
    elif basis.shape[1] == n_signal_features:
        basis_fc = basis.T
    else:
        raise RuntimeError(
            "subspaceBasis is incompatible with signal feature axis: "
            f"signal features={n_signal_features}, subspaceBasis shape={basis.shape}."
        )

    pinv_basis = np.linalg.pinv(basis_fc)
    flat = signal.reshape(-1, n_signal_features)
    projected = flat @ pinv_basis.T
    n_components = int(projected.shape[-1])
    projected = projected.reshape(*signal.shape[:-1], n_components)
    print(
        "Subspace projection applied: "
        f"signal features {n_signal_features} -> {n_components} "
        f"using subspaceBasis shape {basis.shape}."
    )
    return projected

filename = sys.argv[1] if len(sys.argv) > 1 else "data/phantom/20260126_094752_csBFFE_RLT_HR_ImDataParamsBMRR_subspace.h5"

if os.path.exists(filename):
    obj = ImDataParamsRelax(filename)
    obj.reformat_sag2tra()
    if should_collapse_dynamics():
        restore_collapsed_dynamics(obj)
    else:
        sig_shape = obj.ImDataParams["signal"].shape if "signal" in obj.ImDataParams else None
        print(
            "Skipping dynamics collapse (POSTPROC_COLLAPSE_DYNAMICS!=1); "
            f"preserving reconstructed feature axis shape={sig_shape}."
        )
    sig = obj.ImDataParams["signal"]
    if sig.ndim != 5 or min(sig.shape[0:3]) <= 1:
        raise RuntimeError(
            "Reconstruction output has degenerate spatial dimensions "
            f"{sig.shape}. Expected a volumetric image before field-mapping."
        )
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
    if int(np.sum(mask)) < 8:
        sig = obj.ImDataParams["signal"][:, :, :, 0, :]
        raise RuntimeError(
            "tissueMask is too small for dynamic 0. "
            f"Signal max magnitude is {float(np.max(np.abs(sig))):.6g}. "
            f"Mask voxel count is {int(np.sum(mask))}. "
            "Field-mapping requires a non-degenerate 3D region."
        )
    obj.Masks["tissueMask"] = mask

    obj.run_fieldmapping(ind_dynamic=[0]) 
    obj.get_wfi_images(fieldmap_ind_dynamic=0)
    obj.save_WFIparams(mat_file=False)

    obj.load_dictionary("src/Files/20241022_dict_caspr_lookLocker_with0deg_31B0.h5")
    obj.RelaxParams["dictionary"]["B1"] = np.array([1.0])
    obj.RelaxParams["dictionary"]["dictionary"] = obj.RelaxParams["dictionary"]["dictionary"][:,np.newaxis,...]

    # Track A fix (2026-09-13, see experiment/TODO.md): the dictionary's "duration"/"angle"/"delay"
    # parameters describe the Look-Locker readout schedule (per-dynamic inversion-recovery timing),
    # NOT the Dixon dual-echo TEs. Feeding TE_s here previously collapsed both echo queries onto the
    # same wrong dictionary entry (angle always 0), producing a physically meaningless match.
    # prepare_dictionary() already reshapes (nte, numPhases) -> nte*numPhases internally, which matches
    # the reconstruction's own dyn-outer/card-inner fold (Preprocessing.jl changeInterleavesToDynamics!),
    # so passing one duration/angle/delay entry per real Look-Locker dynamic reproduces the true
    # 4 dyn x 7 card = 28 feature contract without any block-mean reduction.
    #
    # obj.perform_dictionary_matching() (src/python/interface.py) rebuilds its own `params` dict
    # internally from obj.RelaxParams['TE_s'/'angle_deg'/'delay_s'/'startup_delay_s'/'prof_ordering'],
    # so those must be set here (not just a local `params` dict) or the internal call would silently
    # fall back to the wrong Dixon-TE-based contract again.
    dict_params = obj.RelaxParams["dictionary"]
    for required_key in ("durations", "angle_deg", "delays"):
        if required_key not in dict_params:
            raise RuntimeError(
                f"Dictionary is missing required key '{required_key}' needed to build the "
                "Look-Locker duration/angle/delay contract."
            )

    durations_ms = np.asarray(dict_params["durations"], dtype=np.float64).flatten()
    angle_deg = np.asarray(dict_params["angle_deg"], dtype=np.float64).flatten()
    delays_ms = np.asarray(dict_params["delays"], dtype=np.float64).flatten()

    obj.RelaxParams["TE_s"] = durations_ms * 1e-3
    obj.RelaxParams["angle_deg"] = angle_deg
    obj.RelaxParams["delay_s"] = delays_ms * 1e-3
    obj.RelaxParams["startup_delay_s"] = np.zeros_like(durations_ms)
    obj.RelaxParams["prof_ordering"] = np.zeros_like(durations_ms, dtype=np.int32)

    params = {}
    params["duration"] = durations_ms
    params["angle"] = angle_deg
    params["delay"] = delays_ms
    params["startup_delay"] = np.zeros_like(durations_ms)
    params["prof_ordering"] = np.zeros_like(durations_ms, dtype=np.int32)
    tmp_dict, _, _ = prepare_dictionary(obj.RelaxParams["dictionary"], params, complex_dict=True)
    n_dict_features = int(tmp_dict.shape[-1])
    print(
        "Track A duration/angle/delay contract: "
        f"nte={len(params['duration'])} durations={params['duration'].tolist()} "
        f"angle_deg={params['angle'].tolist()} delays={params['delay'].tolist()} "
        f"-> n_dict_features={n_dict_features}"
    )

    signal_for_matching = obj.WFIparams["water"]

    use_profiles = False
    if "subspaceBasis" not in obj.ImDataParams:
        print("subspaceBasis not found in reconstruction output; falling back to dictionary matching without profiles.")
    else:
        try:
            signal_for_matching = project_signal_to_subspace(
                signal_for_matching,
                obj.ImDataParams["subspaceBasis"],
            )
            use_profiles = True
        except Exception as e:
            import warnings

            warnings.warn(
                "SubspaceBasis is present but projection failed; "
                "falling back to dictionary matching without profiles. "
                f"Reason: {e}",
                stacklevel=1,
            )

    n_signal_features = int(signal_for_matching.shape[-1])
    if not use_profiles and n_signal_features != n_dict_features:
        if should_reduce_signal_features():
            signal_for_matching = reduce_signal_features_to_target(signal_for_matching, n_dict_features)
        else:
            raise RuntimeError(
                "Signal/dictionary feature mismatch and reduction disabled: "
                f"signal={n_signal_features}, dictionary={n_dict_features}."
            )

    obj.perform_dictionary_matching(
        signal=signal_for_matching,
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
    voxel = np.asarray(obj.ImDataParams.get("voxelSize_mm", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(-1)
    if voxel.size < 3 or (not np.all(np.isfinite(voxel[:3]))) or np.any(voxel[:3] <= 0):
        voxel = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    else:
        voxel = voxel[:3]
    # Arrays are already reoriented by reformat_sag2tra(); avoid a second spatial remap here.
    export_voxel = [float(voxel[0]), float(voxel[1]), float(voxel[2])]
    target_shape = None
    if should_align_reference_grid():
        try:
            import nibabel as nib

            ref_path = Path("data/phantom/reference/T1_ms.nii")
            if ref_path.is_file():
                target_shape = nib.load(str(ref_path)).shape
                print(f"Reference-grid export enabled: target_shape={target_shape}")
        except Exception as e:
            print(f"Reference-grid export skipped: {e}")

    types = ["T1_ms", "T2_ms", "PD"]
    for image_type in types:
        arr = np.abs(obj.RelaxParams[image_type])
        if image_type == "PD":
            # Track C fix (2026-09-13, see experiment/TODO.md): PD is fit as a raw
            # signal-amplitude scale factor and is dominated by coil-sensitivity/
            # receive-intensity shading rather than tissue contrast. Strip the
            # slow-varying bias field via normalized-convolution homomorphic
            # correction, restricted to the same relaxometry tissue mask.
            relaxometry_mask = np.abs(obj.RelaxParams["T1_ms"]) > 0
            arr = homomorphic_bias_correct(arr, relaxometry_mask, sigma=8)
        arr = np.transpose(arr, [1, 0, 2])
        if target_shape is not None and tuple(arr.shape) != tuple(target_shape):
            arr = resize_nn_3d(arr, target_shape)
        obj.export_Array2nii(
            arr,
            f"{folder_path}/{image_type}.nii",
            voxel_size_mm=export_voxel,
        )

    # Track D fix (2026-09-13, see experiment/TODO.md): reuse the same relaxometry
    # tissue mask already computed for T1_ms/T2_ms via obj.set_relaxometry_mask(10)
    # (a PD-percentile threshold mask) instead of exporting the raw unmasked fat
    # magnitude, which was ~93% nonzero (background included) vs the reference's
    # ~20% nonzero. Offline validation showed this more than doubles fat NCC.
    fat_mag = np.abs(obj.WFIparams["fat"][:, :, :, 0])
    relaxometry_mask = np.abs(obj.RelaxParams["T1_ms"]) > 0
    fat_masked = np.zeros_like(fat_mag)
    fat_masked[relaxometry_mask] = fat_mag[relaxometry_mask]
    fat_arr = np.transpose(fat_masked, [1, 0, 2])
    if target_shape is not None and tuple(fat_arr.shape) != tuple(target_shape):
        fat_arr = resize_nn_3d(fat_arr, target_shape)
    obj.export_Array2nii(
        fat_arr,
        f"{folder_path}/coeff1_fat.nii",
        voxel_size_mm=export_voxel,
    )
else:
    print("File '{filename}' not found! Please reconstruct data first.")