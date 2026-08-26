import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
import copy
import warnings

def dictionary_matching(signal, dictionary, params, mask, tmp_dict, tmp_psf, plot_signal=None, 
                        plot_landscape=None, complex_dict=True, compute_pd=True):
    real_signal = np.reshape(signal, [int(np.prod(signal.shape[:-1])),
                                      signal.shape[-1]])
    real_signal_test = copy.deepcopy(real_signal)
    nte = real_signal.shape[-1]

    l2_signal = np.sqrt(np.sum(np.square(np.abs(real_signal)), axis=-1))
    # l2_signal = np.max(np.abs(real_signal), axis=-1)
    real_signal /= l2_signal[:, np.newaxis]

    if 'fieldmap' in params.keys():
        fieldmap = params['fieldmap'][:, np.newaxis]
        fieldmap_dict = dictionary['B0s'].flatten()[:, np.newaxis]

        fieldmap_ind = np.argmin(njit_rmse_prange_dict(fieldmap, fieldmap_dict),
                                 axis=-1)
        tmp_dict = np.reshape(tmp_dict, [tmp_dict.shape[0], int(np.prod(tmp_dict.shape[1:-1])), tmp_dict.shape[-1]])
        tmp_dict = tmp_dict.astype(np.complex64)

        n_dict_features = tmp_dict.shape[-1]
        n_signal_features = real_signal.shape[-1]
        if n_signal_features != n_dict_features:
            if n_signal_features > n_dict_features:
                action = f"truncated to the first {n_dict_features} features"
                accuracy_note = (
                    "Truncation is physically valid only if both the signal and dictionary "
                    "subspace bases share the same SVD ordering (e.g. the signal was "
                    f"reconstructed with {n_signal_features} subspace components and "
                    f"subspaceBasis stores only the leading {n_dict_features})."
                )
            else:
                action = f"zero-padded from {n_signal_features} to {n_dict_features} features"
                accuracy_note = (
                    "Zero-padding is unlikely to produce physically accurate results. "
                    "Ensure the signal and dictionary are in the same subspace."
                )
            warnings.warn(
                f"Signal feature dimension ({n_signal_features}) does not match dictionary "
                f"feature dimension ({n_dict_features}). The signal will be {action}. "
                f"{accuracy_note} "
                f"The root cause is typically that the reconstruction used a different number "
                f"of subspace components than what is stored in subspaceBasis.",
                stacklevel=3
            )
            if n_signal_features > n_dict_features:
                real_signal = real_signal[:, :n_dict_features]
                real_signal_test = real_signal_test[:, :n_dict_features]
            else:
                pad_width = n_dict_features - n_signal_features
                real_signal = np.pad(real_signal, ((0, 0), (0, pad_width)))
                real_signal_test = np.pad(real_signal_test, ((0, 0), (0, pad_width)))
            # Re-normalise after reshaping the signal
            l2_signal_adj = np.sqrt(np.sum(np.square(np.abs(real_signal)), axis=-1))
            l2_signal_adj[l2_signal_adj == 0] = 1.0
            real_signal = real_signal / l2_signal_adj[:, np.newaxis]

        # err = np.zeros((real_signal.shape[0], tmp_dict.shape[1]))
        # for i in range(len(fieldmap_dict)):
        #     mask = (fieldmap_ind == i)
        #     err[mask, :] = np.abs(np.inner(np.conj(real_signal[mask, :]), tmp_dict[i,:,:]))
        # min_err = np.argmax(err, axis=-1)
        err = njit_rmse_prange_voxels(np.conj(real_signal), tmp_dict, fieldmap_ind)
        min_err = np.argmax(err, axis=-1)
        # min_err = np.argmin(err, axis=-1)

        b1, t1, t2 = np.unravel_index(min_err, dictionary['dictionary'].shape[1:4])
        # Dot Product
        dot_product = np.max(err, axis=-1)

        pd = np.zeros(real_signal.shape[0])
        if compute_pd:
            if complex_dict:
                # tmp_dict = tmp_dict.astype(np.complex64)
                pd = compute_pd_numba(tmp_dict, fieldmap_ind, min_err, real_signal_test)
            else:
                tmp_dict = tmp_dict.astype(np.float32)
                pd = compute_pd_numba_float(tmp_dict, fieldmap_ind, min_err, real_signal_test)
        else:
            pd = np.zeros(real_signal_test.shape[0])
    else:
        if plot_signal:
            _plot_signal(real_signal, tmp_dict, mask, dictionary, plot_signal)

        tmp_dict = np.reshape(tmp_dict, [int(np.prod(tmp_dict.shape[:-1])),
                                         tmp_dict.shape[-1]])

        # print(tmp_dict[1000,:])

        err = np.inner(real_signal, tmp_dict)
        min_err = np.argmax(err, axis=-1)
        # err = njit_rmse_prange_dict(real_signal, tmp_dict)
        # min_err = np.argmin(err, axis=-1)

        if plot_landscape:
            _plot_landscape(err, mask, dictionary, plot_landscape)

        #if len(dictionary['dictionary'].shape) == 3:
        t1, t2 = np.unravel_index(min_err, dictionary['dictionary'].shape[:2])
        pd = np.zeros(real_signal.shape[0])
        # if compute_pd:
        #     for i in range(pd.shape[0]):
        #         a, _, _, _ = np.linalg.lstsq(tmp_dict[min_err[i],:,np.newaxis], real_signal_test[i,:], rcond=None)
        #         pd[i] = a
        # else:
        #     b1, t1, t2 = np.unravel_index(min_err, dictionary['dictionary'].shape[:3])

    #pd = np.min(np.divide(real_signal_test, tmp_dict[min_err, :]), axis=-1)
    #pd = np.diag(np.inner(real_signal_test, tmp_dict[min_err, :]))
    # for i in range(pd.shape[0]):
    #     a, _, _, _ = np.linalg.lstsq(tmp_dict[min_err[i],:,np.newaxis], real_signal_test[i,:], rcond=None)
    #     pd[i] = a
    #
    # real_signal_test = real_signal_test.T
    # pd = np.linalg.inv(real_signal_test.T @ real_signal_test) @ real_signal_test.T @ tmp_dict[min_err,:]
    # print(pd)
    # pd = l2_signal / l2_dict[min_err]
    t1map = dictionary['T1s'].flatten()[t1]
    t2map = dictionary['T2s'].flatten()[t2]
    if 'b1' in locals():
        b1map = dictionary['B1'].flatten()[b1]
    else:
        b1map = None
    if 'dot_product' not in locals():
        dot_product = None
    if tmp_psf is not None:
        psf = tmp_psf[fieldmap_ind, t1, t2, :, :]
    else:
        psf = None
    return t1map, t2map, b1map, pd, psf, dot_product

def prepare_dictionary(dictionary, params, complex_dict=True):
    nte = len(params['duration'])
    if complex_dict:
        # tmp_dict = np.real(dictionary['dictionary']*np.exp(-1j*np.angle(dictionary['dictionary'][...,1,:][...,np.newaxis,:])))
        # tmp_dict = (dictionary['dictionary']*np.exp(-1j*np.angle(dictionary['dictionary'][...,1,0][...,np.newaxis,np.newaxis])))
        tmp_dict = copy.deepcopy(dictionary['dictionary'])
    else:
        tmp_dict = np.abs(dictionary['dictionary'])
    if 'psf' in dictionary.keys():
        tmp_psf = dictionary['psf']
    else:
        tmp_psf = None
    # if "kz_profiles" in dictionary.keys():
    #     tmp_dict = tmp_dict[..., 0]
    if 'angle_deg' in dictionary and 'delays' in dictionary and not 'startup_delays' in dictionary:
        print(tmp_dict.shape)
        dict_param = np.concatenate((dictionary['durations'][:, np.newaxis],
                                     dictionary['angle_deg'][:, np.newaxis],
                                     dictionary['delays'][:, np.newaxis]), axis=-1)
        # dict_te_series = np.zeros((np.concatenate((list(tmp_dict.shape[:-1]),
        #                                            [nte]))))
        dict_te_series = np.zeros((np.concatenate((list(tmp_dict.shape[:-3]),
                                                   [nte, tmp_dict.shape[-2]]))), dtype=np.complex64)
        if tmp_psf is not None:
            psf_te_series = np.zeros((np.concatenate((list(tmp_psf.shape[:-2]),
                                                      [nte, tmp_psf.shape[-1]]))))

        prof_ord_indices = [0, tmp_dict.shape[-1]//2 , -1]
        for i in range(nte):
            ind = np.argwhere((np.abs(dict_param - np.array([params['duration'][i],
                                                             params['angle'][i],
                                                             params['delay'][i]]))<1e-4).all(axis=1) == True)
            print(dict_param)
            print(params["duration"][i])
            print(params["angle"][i])
            print(params["delay"][i])
            dict_te_series[..., i, :] = tmp_dict[..., ind.flatten()[0], :, prof_ord_indices[params["prof_ordering"][i]]]
            print(dict_te_series.shape)
            if tmp_psf is not None:
                psf_te_series[..., i, :] = tmp_psf[..., ind.flatten()[0], :]
        tmp_dict = np.reshape(dict_te_series, np.concatenate((list(tmp_dict.shape[:-3]), [nte*tmp_dict.shape[-2]])))
        print(tmp_dict.shape)
        if tmp_psf is not None:
            tmp_psf = psf_te_series
    elif 'angle_deg' in dictionary and 'delays' in dictionary and 'startup_delays' in dictionary \
         and 'prof_ordering' in dictionary:
        dict_param = np.concatenate((dictionary['durations'][:, np.newaxis],
                                     dictionary['angle_deg'][:, np.newaxis],
                                     dictionary['delays'][:, np.newaxis],
                                     dictionary['startup_delays'][:, np.newaxis]), axis=-1)
        if complex_dict:
            dict_te_series = np.zeros((np.concatenate((list(tmp_dict.shape[:-2]),
                                                       [nte]))), dtype=np.complex64)
        else:
            dict_te_series = np.zeros((np.concatenate((list(tmp_dict.shape[:-2]),
                                                    [nte]))))
        if tmp_psf is not None:
            psf_te_series = np.zeros((np.concatenate((list(tmp_psf.shape[:-2]),
                                                      [nte, tmp_psf.shape[-1]]))))
        prof_ord_indices = [0, tmp_dict.shape[-1]//2 , -1]
        for i in range(nte):
            ind = np.argwhere((np.abs(dict_param - np.array([params['duration'][i],
                                                             params['angle'][i],
                                                             params['delay'][i],
                                                             params['startup_delay'][i]]))<1e-6).all(axis=1) == True)
            # print(np.abs(dict_param - np.array([params['duration'][i],
            #                                                  params['angle'][i],
            #                                                  params['delay'][i],
            #                                                  params['startup_delay'][i]]))<1e-6).all(axis=1)
            # print(params['duration'][i])
            # print(params['angle'][i])
            # print(params['delay'][i])
            # dict_te_series[..., i] = tmp_dict[..., ind.flatten()[0]]
            # print(ind.flatten())
            dict_te_series[..., i] = tmp_dict[..., ind.flatten()[0], prof_ord_indices[params["prof_ordering"][i]]]
            if tmp_psf is not None:
                psf_te_series[..., i, :] = tmp_psf[..., ind.flatten()[0], :]
        tmp_dict = dict_te_series
        if tmp_psf is not None:
            tmp_psf = psf_te_series
    else:
        mask_durs = np.squeeze((dictionary['durations'] - params['duration']) < 1e-6)
        tmp_dict = tmp_dict[..., mask_durs, 0]
        tmp_psf = None
    # assert tmp_dict.shape[-1] == nte
    l2_dict = np.sqrt(np.sum(np.square(np.abs(tmp_dict)), axis=-1))
    # l2_dict = np.max(tmp_dict, axis=-1)
    tmp_dict /= l2_dict[..., np.newaxis]
    # np.abs(tmp_dict)
    # tmp_dict = (tmp_dict - np.min(tmp_dict, axis=-1)[..., np.newaxis]) / (np.max(tmp_dict, axis=-1)[..., np.newaxis] - np.min(tmp_dict, axis=-1)[..., np.newaxis])
    return tmp_dict, tmp_psf, l2_dict

def prepare_dictionary_profiles(dictionary, params, subspace_basis, complex_dict=True, l2_dict=None):
    nte = len(params['duration'])
    if complex_dict:
        # tmp_dict = np.real(dictionary['dictionary']*np.exp(-1j*np.angle(dictionary['dictionary'][...,0,-1][...,np.newaxis,np.newaxis])))
        # import pdb; pdb.set_trace()
        # tmp_dict = np.real(dictionary['dictionary']*np.exp(-1j*np.angle(dictionary['dictionary'][...,0,0][...,np.newaxis,np.newaxis])))
        # tmp_dict = (dictionary['dictionary']*np.exp(-1j*np.angle(dictionary['dictionary'][...,1,0][...,np.newaxis,np.newaxis])))
        # tmp_dict = copy.deepcopy(dictionary['dictionary'])
        tmp_dict = copy.deepcopy(dictionary['dictionary'])
    else:
        # tmp_dict = np.abs(dictionary['dictionary'])
        tmp_dict = copy.deepcopy(dictionary['dictionary'])
    tmp_dict = copy.deepcopy(dictionary['dictionary'])
    if 'psf' in dictionary.keys():
        tmp_psf = dictionary['psf']
    else:
        tmp_psf = None

    dict_param = np.concatenate((dictionary['durations'][:, np.newaxis],
                                dictionary['angle_deg'][:, np.newaxis],
                                dictionary['delays'][:, np.newaxis]), axis=-1)
    dict_te_series = np.zeros((np.concatenate((list(tmp_dict.shape[:-3]),
                                                [nte, tmp_dict.shape[-2], tmp_dict.shape[-1]]))), dtype=np.complex64)
    if tmp_psf is not None:
        psf_te_series = np.zeros((np.concatenate((list(tmp_psf.shape[:-2]),
                                                    [nte, tmp_psf.shape[-1]]))))
    for i in range(nte):
        ind = np.argwhere((np.abs(dict_param - np.array([params['duration'][i],
                                                            params['angle'][i],
                                                            params['delay'][i]]))<1e-4).all(axis=1) == True)
        dict_te_series[..., i, :, :] = tmp_dict[..., ind.flatten()[0], :, :]
        if tmp_psf is not None:
            psf_te_series[..., i, :] = tmp_psf[..., ind.flatten()[0], :]
    tmp_dict = np.reshape(dict_te_series, np.concatenate((list(tmp_dict.shape[:-3]), [nte*tmp_dict.shape[-2]*tmp_dict.shape[-1]])))
    print(tmp_dict.shape)
    if tmp_psf is not None:
        tmp_psf = psf_te_series

    if l2_dict is not None:
        tmp_dict /= l2_dict[..., np.newaxis]

    tmp_dict = tmp_dict @ subspace_basis
    # #subspace_basis = np.load("/Users/jonathanstelter/Desktop/prep_dict_basis.npy")
    # # subspace_basis = np.load("/Users/jonathanstelter/Desktop/prep_dict_0147_basis.npy")
    # subspace_basis = np.load("/Users/jonathanstelter/Desktop/ISMRM24/prep_dict_0714_B0B1_basis_2.npy")
    # # subspace_basis = np.load("/Users/jonathanstelter/Desktop/prep_dict_0714_basis.npy")
    # # # # import pdb; pdb.set_trace()
    # tmp_dict = tmp_dict @ subspace_basis[:,:4]

    # tmp_dict = tmp_dict[:,10:11, :, :, :]
    
    if l2_dict is None:
        l2_dict = np.sqrt(np.sum(np.square(np.abs(tmp_dict)), axis=(-1))) #/ (tmp_dict.shape[-1]*tmp_dict.shape[-2])
        # l2_dict = np.sqrt(np.sum(np.square(np.abs(tmp_dict)), axis=(-1))) #/ (tmp_dict.shape[-1]*tmp_dict.shape[-2])
        # l2_dict = np.max(tmp_dict, axis=(-1,-2))
        tmp_dict /= l2_dict[..., np.newaxis]
    # tmp_dict = (tmp_dict - np.min(tmp_dict, axis=-1)[..., np.newaxis]) / (np.max(tmp_dict, axis=-1)[..., np.newaxis] - np.min(tmp_dict, axis=-1)[..., np.newaxis])
    return tmp_dict, tmp_psf

def forward_simulation(t1_map, t2_map, b1_map, dictionary, params, tmp_dict, tmp_psf):
    t1_dict = np.repeat(dictionary['T1s'][:, np.newaxis], t1_map.shape[0], axis=-1)
    t2_dict = np.repeat(dictionary['T2s'][:, np.newaxis], t2_map.shape[0], axis=-1)
    b1_dict = np.repeat(dictionary['B1'][:, np.newaxis], t2_map.shape[0], axis=-1)
    t1_map = np.repeat(t1_map[np.newaxis,:], t1_dict.shape[0], axis=0)
    t2_map = np.repeat(t2_map[np.newaxis,:], t2_dict.shape[0], axis=0)
    b1_map = np.repeat(b1_map[np.newaxis,:], b1_dict.shape[0], axis=0)

    t1 = np.argmin(np.abs(t1_map-t1_dict), axis=0)
    t2 = np.argmin(np.abs(t2_map-t2_dict), axis=0)
    b1 = np.argmin(np.abs(b1_map-b1_dict), axis=0)
    if 'fieldmap' in params.keys():
        fieldmap = params['fieldmap'][:, np.newaxis]
        fieldmap_dict = dictionary['B0s'].flatten()[:, np.newaxis] # why not as t1/t2

        fieldmap_ind = np.argmin(njit_rmse_prange_dict(fieldmap, fieldmap_dict), axis=-1)
        if tmp_psf is not None:
            psf = tmp_psf[fieldmap_ind, t1, t2, :, :]
        else:
            psf = None
        signal = tmp_dict[fieldmap_ind, b1, t1, t2, :]
    else:
        raise NotImplementedError
    return signal, psf


@njit(parallel=True)
def njit_rmse_prange_dict(real_signal, tmp_dict):
    nte = real_signal.shape[-1]
    err = np.zeros((real_signal.shape[0], tmp_dict.shape[0]))
    for i in prange(tmp_dict.shape[0]):
        simulated_signal = tmp_dict[i]
        err[:, i] = 1 / nte * (np.sum(np.square(real_signal -
                                                simulated_signal), axis=-1))
    return err

@njit(parallel=True)
def njit_rmse_prange_voxels(real_signal, tmp_dict, fieldmap_ind):
    nte = real_signal.shape[-1]
    err = np.zeros((real_signal.shape[0], tmp_dict.shape[1]))
    for i in prange(real_signal.shape[0]):
        real_signal_voxel = real_signal[i]
        simulated_signal_voxel = tmp_dict[fieldmap_ind[i]]
        dot_product = simulated_signal_voxel @ real_signal_voxel.conj()
        err[i, :] = np.abs(dot_product)
        # err[i, :] = 1 / nte * (np.sum(np.square(np.abs(real_signal_voxel -
        #                                         simulated_signal_voxel)), axis=-1))
    return err

    err = np.zeros((real_signal.shape[0], tmp_dict.shape[1]), dtype=np.float32)
    for i in prange(fieldmap_dict_len):
        mask = (fieldmap_ind == i)
        
        for voxel_idx in range(real_signal.shape[0]):
            if mask[voxel_idx]:  # Check if the voxel is part of the mask
                # Calculate the inner product for this voxel and the current dictionary entry
                real_part = real_signal[voxel_idx, :]
                tmp_entry = tmp_dict[i, :, :]
                
                # Compute the dot product of the conjugate of `real_part` and `tmp_entry`
                dot_product = np.sum(real_part.conj() * tmp_entry, axis=0)
                err[voxel_idx, :] = np.abs(dot_product)
                
    return err
# @njit(parallel=True)
# def compute_pd_numba(tmp_dict, fieldmap_ind, min_err, real_signal):
#     pd = np.zeros(real_signal.shape[0], dtype=np.complex64)
#     for i in prange(pd.shape[0]):
#         a, _, _, _ = np.linalg.lstsq(tmp_dict[fieldmap_ind[i], min_err[i], :, np.newaxis],
#                                        real_signal[i, :], rcond=None)
#         pd[i] = a 
#     return pd

@njit
def custom_lstsq(a, b):
    ata = np.dot(a.T, a)
    atb = np.dot(a.T, b)
    return np.linalg.solve(ata, atb)

# @njit(parallel=True)
def compute_pd_numba(tmp_dict, fieldmap_ind, min_err, real_signal_test):
    num_signals = real_signal_test.shape[0]
    pd = np.zeros(num_signals, dtype=np.complex64) 
    
    for i in prange(num_signals):
        # We need to avoid using np.newaxis, so we create the necessary arrays with the correct shapes manually.
        dict_slice = tmp_dict[fieldmap_ind[i], min_err[i]].reshape(-1, 1)  # Convert to a 2D array.
        signal = real_signal_test[i, :]  # This should naturally be a 1D array.

        # Solve the least squares problem.
        a = custom_lstsq(dict_slice, signal)

        # Store the result.
        pd[i] = a[0] 
    return pd

@njit(parallel=True)
def compute_pd_numba_float(tmp_dict, fieldmap_ind, min_err, real_signal_test):
    num_signals = real_signal_test.shape[0]
    pd = np.zeros(num_signals) 
    
    for i in prange(num_signals):
        # We need to avoid using np.newaxis, so we create the necessary arrays with the correct shapes manually.
        dict_slice = tmp_dict[fieldmap_ind[i], min_err[i]].reshape(-1, 1)  # Convert to a 2D array.
        signal = real_signal_test[i, :]  # This should naturally be a 1D array.

        # Solve the least squares problem.
        a = custom_lstsq(dict_slice, signal)

        # Store the result.
        pd[i] = a[0] 
    return pd


def _plot_landscape(rmse, mask, dictionary, plot_landscape):
    rmse3d = np.zeros((mask.shape[0], mask.shape[1],
                        rmse.shape[-1]))
    rmse3d[mask] = rmse
    if len(plot_landscape) == 2:
        clim = plot_landscape[1]
    else:
        clim = np.log([np.min(rmse3d[plot_landscape[0][0], plot_landscape[0][1]]),
                        np.max(rmse3d[plot_landscape[0][0], plot_landscape[0][1]])])
    plt.figure()
    print(dictionary['dictionary'].shape)
    if len(dictionary['dictionary'].shape) == 4:
        plt.imshow(np.reshape(np.log(rmse3d[plot_landscape[0][0], plot_landscape[0][1]]), (dictionary['dictionary'].shape[0]*dictionary['dictionary'].shape[1],
                                                                                            dictionary['dictionary'].shape[2])),
                vmin=clim[0], vmax=clim[1], extent=[dictionary['T2s'][0],
                                                    dictionary['T2s'][-1],
                                                    dictionary['T1s'][-1],
                                                    dictionary['T1s'][0]],
                aspect='auto')
    else:
        plt.imshow(np.reshape(np.log(rmse3d[plot_landscape[0][0], plot_landscape[0][1]]), (dictionary['dictionary'].shape[0],
                                                                dictionary['dictionary'].shape[1])),
                vmin=clim[0], vmax=clim[1], extent=[dictionary['T2s'][0],
                                                    dictionary['T2s'][-1],
                                                    dictionary['T1s'][-1],
                                                    dictionary['T1s'][0]],
                aspect='auto')
    plt.ylabel('T1 (ms)')
    plt.xlabel('T2 (ms)')
    plt.colorbar()
    plt.show()

def _plot_signal(real_signal, tmp_dict, mask, dictionary, plot_signal):
    nte = real_signal.shape[-1]
    plt.figure()
    plt.scatter(np.arange(nte), tmp_dict[np.abs(dictionary['T1s'] - plot_signal[1][0]) < 0.001,
                                         np.abs(dictionary['T2s'] - plot_signal[1][1]) < 0.001], label='dictionary')
    sig3d = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], nte))
    sig3d[mask] = real_signal
    plt.scatter(np.arange(nte), sig3d[plot_signal[0][0], plot_signal[0][1], plot_signal[0][2], :], label='signal')
    plt.legend()
    plt.show()
