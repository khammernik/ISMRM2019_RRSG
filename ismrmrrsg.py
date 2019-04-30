#!/usr/bin/env python3
"""
Copyright (C) 2019 Kerstin Hammernik <hammernik at icg dot tugraz dot at>
Institute of Computer Graphics and Vision, Graz University of Technology
https://www.tugraz.at/institute/icg/research/team-pock/

Code submission for the Reproducible Research Study Group (RRSG) at ISMRM 2019
https://blog.ismrm.org/2019/04/02/ismrm-reproducible-research-study-group-2019-reproduce-a-seminal-paper-initiative/

Reproduction of experiments in [1]

[1] Pruessmann, K. P.; Weiger, M.; Boernert, P. and Boesiger, P.
Advances in sensitivity encoding with arbitrary k-space trajectories.
Magn Reson Med 46: 638-651 (2001)
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import primaldualtoolbox
import medutils
import os

def loadDataset(fname):
    """ Load the dataset. 
        The datasets for the RRSG can be downloaded from http://wwwuser.gwdg.de/~muecker1/rrsg_challenge.zip
    :param fname: input filename
    :return: kspace, trajectory, config for gpuNUFFT
    """
    h5_dataset = h5py.File(fname, 'r')
    h5_dataset_kspace_name = list(h5_dataset.keys())[0]
    h5_dataset_trajectory_name = list(h5_dataset.keys())[1]

    trajectory = h5_dataset.get(h5_dataset_trajectory_name).value
    kspace = h5_dataset.get(h5_dataset_kspace_name).value[0]
    h5_dataset.close()
    
    if 'rawdata_brain_radial_96proj_12ch.h5' in fname:
        osf = 512/300
        config = {'osf' : osf, 'img_dim' : int(kspace.shape[0]/osf)}
    elif 'rawdata_heart_radial_55proj_34ch.h5' in fname:
        osf = 4/3
        config = {'osf' : osf, 'img_dim' : int(kspace.shape[0]/osf)}
    else:
        config = {'osf' : 2, 'img_dim' : kspace.shape[0]//2}
    
    return kspace, trajectory, config

def prepareTrajectory(trajectory, save_fig=False, save_dir='./'):
    """ Prepare the trajectory and normalize it between [-0.5, 0.5] required for gpuNUFFT
    :param trajectory: input trajectory as loaded from the dataset
    :param save_fig: save trajectory and dcf
    :param save_dir: directory where trajectory and dcf should be saved.
    :return: trajectory (column), dcf (column)
    """
    # extract trajectory
    trajectory_x = trajectory.copy()[0]
    trajectory_y = trajectory.copy()[1]

    # compute dcf
    dcf = np.abs(trajectory_x + 1j*trajectory_y)
    scale = 2*np.max(dcf)
    dcf /= (scale/2)
    trajectory_x /= scale
    trajectory_y /= scale
    
    nFE, nSpokes = dcf.shape
    #dcf *= np.pi/(4*nSpokes)

    plt.figure()
    plt.plot(trajectory_x, trajectory_y)
    plt.title('trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    if save_fig:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f'{save_dir}/trajectory.png', bbox_inches='tight')
    
    plt.figure()
    plt.plot(np.linspace(-0.5, 0.5, nFE), dcf)
    plt.title('dcf')
    plt.xlabel('x')
    plt.ylabel('y')
    if save_fig:
        plt.savefig(f'{save_dir}/dcf.png', bbox_inches='tight')

    # re-arange data to be used in primaldualtoolbox
    trajectory_col = np.array([trajectory_y.flatten(), trajectory_x.flatten()])
    dcf_col = dcf.flatten()[np.newaxis,...]
    return trajectory_col, dcf_col

def regridRadialData(kspace, trajectory_col, dcf_col, nufft_config, use_cg=False):
    """ Regrid radial data
    :param kspace: input k-space
    :param trajectory_col: trajectory (column)
    :param dcf_col: dcf (column)
    :param nufft_config: gpuNUFFT config
    :param use_cg: Use CG for regridding instead of computing the simple adjoint
    :return: regridded k-space
    """
    # setup non-uniform operator
    op_nufft = primaldualtoolbox.mri.MriRadialSinglecoilOperator(nufft_config)

    # set operator constants. This has to be done in the right order.
    op_nufft.setTrajectory(trajectory_col)
    op_nufft.setDcf(dcf_col)

    # regridding to cartesian grid
    img_dim = nufft_config['img_dim']
    nFE, nSpokes, nCh = kspace.shape
    
    img = np.zeros((img_dim, img_dim, nCh), dtype=np.complex64)

    # reshape kspace to be used in primaldualtoolbox
    kspace_reshape = np.reshape(kspace, (1, nFE*nSpokes, nCh), 'C')

    if use_cg:
        # re-grid single channels using CG optimizer. Pre-weight rawdata with sqrt(dcf)
        optimizer = medutils.optimization.CgSenseReconstruction(op_nufft, max_iter=50, alpha=0, tol=1e-30)
        for ch in range(nCh):
            img[...,ch] = optimizer.solve(np.ascontiguousarray(kspace_reshape[...,ch]*np.sqrt(dcf_col)))
    else:
        # re-grid single channels. Pre-weight rawdata with sqrt(dcf)
        for ch in range(nCh):
            img[...,ch] = op_nufft.adjoint(np.ascontiguousarray(kspace_reshape[...,ch]*np.sqrt(dcf_col)))

    # Apply Fourier transform to obtain a set of Cartesian regridded kspace for coil sensitivity estimation
    kspace_regridded = medutils.mri.fft2c(img, axes=(0,1))
    # clean-up
    del op_nufft
    return kspace_regridded
    
def plot(fname, ic, dc):
    """ Generate required plots and images.
    :param fname: input filename
    :param ic: intensity compensation
    :param dc: density compensation
    """
    if 'rawdata_brain_radial_96proj_12ch.h5' in fname:
        # convergence plots
        legend = []

        fig_tol = plt.figure()
        fig_error = plt.figure()
        ax_tol = fig_tol.add_subplot(1,1,1)
        ax_err = fig_error.add_subplot(1,1,1)

        output = []
        for R in range(1,5):
            output_dir_acc = fname.split('.h5')[0] + f'_R{R}'
            h5_results = h5py.File(f'{output_dir_acc}/results.h5', 'r')
            note = f"{'_ic' if ic else ''}{'_dc' if dc else ''}"
            cgsense_series = h5_results[f'cgsense{note}'].value
            img_regridded = h5_results[f'img_regridded'].value
            data_tol = h5_results[f'tol{note}'].value
            legend.append(f'R={R}')
            output.append((cgsense_series[-1], cgsense_series[0], img_regridded[...,0], data_tol.shape[0]))
            h5_results.close()

            if R==1:
                reference = cgsense_series[-1]

            data_err = np.linalg.norm(cgsense_series - reference, axis=(1,2))**2/np.linalg.norm(reference)**2
            ax_tol.semilogy(np.arange(1, len(data_tol)+1), data_tol)
            if R > 1:
                ax_err.semilogy(np.arange(1, len(data_err)+1), data_err)

        ax_tol.legend(legend)
        ax_tol.set_xlabel('Iterations')
        ax_tol.set_ylabel('log$_{10}\delta$')
        ax_tol.set_title('Convergence of the $\delta$ criterion')
        ax_err.legend(legend[1:])
        ax_err.set_xlabel('Iterations')
        ax_err.set_ylabel('log$_{10}\Delta$')
        ax_err.set_title('Convergence of the true image error $\Delta$')
        fig_tol.savefig('brain_convergence_tol.png', bbox_inches='tight', pad_inches = 0, dpi=600)
        fig_error.savefig('brain_convergence_imageerror.png', bbox_inches='tight', pad_inches = 0, dpi=600)

        # image plots
        fig = plt.figure(figsize=(3,4))
        ax = [fig.add_subplot(4,3,i+1) for i in range(12)]

        def draw(ax, img, text):
            settings = {'verticalalignment': 'bottom',

                        'horizontalalignment':'right',
                        'transform': ax.transAxes,
                        'color':'white',
                        'fontsize':'5'}
            ax.imshow(np.abs(img), cmap='gray')
            ax.text(1, 0, str(text), **settings)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        for R in range(1,5):
            img, img_init, coil_img, stop_iter = output[R-1]
            draw(ax[R*3-3], coil_img, 1)
            draw(ax[R*3-2], img_init, 1)
            draw(ax[R*3-1], img, stop_iter)
            ax[R*3-3].text(-0.02, 0.5, f'R={R}', ha='right', va='center', fontsize='6', transform=ax[R*3-3].transAxes,                   
                           rotation='vertical')

        ax[0].text(0.5, 1.02, 'Single coil', ha='center', va='bottom', fontsize='6', transform=ax[0].transAxes)
        ax[1].text(0.5, 1.02, 'Initial', ha='center', va='bottom', fontsize='6', transform=ax[1].transAxes)
        ax[2].text(0.5, 1.02, 'Final', ha='center', va='bottom', fontsize='6', transform=ax[2].transAxes)

        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        fig.savefig('brain_comparison.png', bbox_inches='tight', pad_inches = 0, dpi=1200)
        
    if 'rawdata_heart_radial_55proj_34ch.h5' in fname:
        fig = plt.figure(figsize=(4,1), frameon=False)
        ax = [fig.add_subplot(1,4,i+1) for i in range(4)]

        def draw(ax, img, text):
            settings = {'verticalalignment': 'bottom',

                        'horizontalalignment':'right',
                        'transform': ax.transAxes,
                        'color':'white',
                        'fontsize':'5'}
            ax.imshow(np.abs(img), cmap='gray')
            ax.text(1, 0, str(text), **settings)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        proj_list = [55, 33, 22, 11]
        output = []
        legend = []
        for num_proj in proj_list:
            output_dir_acc = fname.split('.h5')[0] + f'_proj{num_proj}'
            h5_results = h5py.File(f'{output_dir_acc}/results.h5', 'r')
            note = f"{'_ic' if ic else ''}{'_dc' if dc else ''}"
            cgsense_series = h5_results[f'cgsense{note}'].value
            img_regridded = h5_results[f'img_regridded'].value
            data_tol = h5_results[f'tol{note}'].value
            legend.append(f'{num_proj} proj')
            output.append((cgsense_series[-1], cgsense_series[0], img_regridded[...,0], data_tol.shape[0]))
            h5_results.close()

        for idx in range(0, 4):
            img, img_init, coil_img, stop_iter = output[idx]
            draw(ax[idx], img, stop_iter)
            ax[idx].text(0.5, 1.02, legend[idx], ha='center', va='bottom', fontsize='6', transform=ax[idx].transAxes)

        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        fig.savefig('heart_comparison.png', bbox_inches='tight', pad_inches = 0, dpi=1200)
