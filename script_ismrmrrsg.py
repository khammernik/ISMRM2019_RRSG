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
import matplotlib
matplotlib.use("Agg")

import ismrmrrsg
import medutils
import medutils.visualization as vis
import numpy as np
import primaldualtoolbox
import matplotlib.pyplot as plt
import h5py

import urllib.request
import os
import zipfile
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['brain', 'heart'])
args = parser.parse_args()

if not os.path.exists('./data'):
    print('Download RRSG challenge data...')

    url = 'http://wwwuser.gwdg.de/~muecker1/rrsg_challenge.zip'  
    urllib.request.urlretrieve(url, 'rrsg_challenge.zip')
    with zipfile.ZipFile("rrsg_challenge.zip","r") as zip_ref:
        zip_ref.extractall(".")
    os.remove('./rrsg_challenge.zip')
    os.rename('./rrsg_challenge', './data')
    
fname = './data/rawdata_heart_radial_55proj_34ch.h5' if args.dataset == 'heart' else './data/rawdata_brain_radial_96proj_12ch.h5'

ic = True  # intensity compensation
dc = True  # density compensation
alpha = 0.2 # Tikhonov regularization
use_cg_regrid = False

kspace, trajectory, nufft_config = ismrmrrsg.loadDataset(fname)
nFE, nSpokes, nCh = kspace.shape

output_dir = fname.split('.h5')[0].split('/')[-1]
print(f'Save all plots to {output_dir}...')

print(f'Shape of trajectory: {trajectory.shape}')
print(f'Shape of radial kspace: {kspace.shape}')

# setup nufft config
nufft_config['sector_width'] = 8
nufft_config['kernel_width'] = 3
print(f'gpuNufft config: {nufft_config}')

# show kspace
vis.show(kspace, transpose=(2,0,1), title="k-space of coil", logscale=True)

# save kspace of a single coil
plot_coil=0
vis.save(kspace[...,plot_coil], f'{output_dir}/kspace_coil{plot_coil}.png', logscale=True, cmap='gray')

# prepare trajectory
trajectory_col, dcf_col = ismrmrrsg.prepareTrajectory(trajectory, save_dir=output_dir, save_fig=True)

# regrid radial kspace
kspace_regridded = ismrmrrsg.regridRadialData(kspace, trajectory_col, dcf_col, nufft_config, use_cg=use_cg_regrid)

# save regridded rss reconstruction
img_regridded = medutils.mri.ifft2c(kspace_regridded, axes=(0,1))
vis.save(medutils.visualization.flip(img_regridded[...,plot_coil]), f'{output_dir}/img_coil{plot_coil}.png', cmap='gray')
img_regridded_rss = medutils.mri.rss(img_regridded, coil_axis=-1)
vis.save(medutils.visualization.flip(img_regridded_rss), f'{output_dir}/rss.png', cmap='gray')

# compute coil sensitivity maps using espirit
# using the -I flag, the smaps are not intensity normalized
smaps = medutils.bart(1, 'ecalib -d0 -m1 -I', kspace_regridded[None,...])[0]
vis.show(medutils.visualization.flip(smaps, axes=(0,1)), transpose=(2,0,1), title='sensitivity map')
vis.save(medutils.visualization.flip(smaps[...,plot_coil]), f'{output_dir}/smaps_coil{plot_coil}.png', cmap='gray')

# intensity compensation map
icmap = medutils.mri.rss(smaps, coil_axis=-1)
vis.show(medutils.visualization.flip(icmap), title='intensity compensation map')

# prepare smaps for primaldualtoolbox
smaps = np.ascontiguousarray(np.transpose(smaps, (2,0,1)))

# generate reconstructions for brain
if 'rawdata_brain_radial_96proj_12ch.h5' in fname:
    print(f'intensity_compensation={ic}, density_compenation={dc}')
    for R in range(1,5):
        kspace_acc = kspace[...,::R,:]
        nFE, nSpokes, nCH = kspace_acc.shape
        trajectory_acc = trajectory[...,::R]

        output_dir_acc = fname.split('.h5')[0] + f'_R{R}'

        # prepare trajectory
        trajectory_col, dcf_col = ismrmrrsg.prepareTrajectory(trajectory_acc, save_dir=output_dir_acc, save_fig=True)
        kspace_radial = np.ascontiguousarray(np.reshape(np.transpose(kspace_acc, (2,0,1)), (nCh, nFE*nSpokes)))*np.sqrt(dcf_col)

        kspace_regridded = ismrmrrsg.regridRadialData(kspace_acc, trajectory_col, dcf_col, nufft_config, use_cg=use_cg_regrid)
        
        # show regridded rss reconstruction
        img_regridded = medutils.mri.ifft2c(kspace_regridded, axes=(0,1))

        # setup radial operator
        op = primaldualtoolbox.mri.MriRadialOperator(nufft_config)
        op.setTrajectory(trajectory_col)
        op.setDcf(dcf_col if dc else np.ones_like(dcf_col))
        op.setCoilSens(np.nan_to_num(smaps/icmap) if ic else smaps)
        
        # CG SENSE Reconstruction
        optimizer = medutils.optimization.CgSenseReconstruction(op=op, max_iter=100, alpha=alpha, tol=1e-20)
        cgsense, acc = optimizer.solve(kspace_radial, return_tol=True, return_series=True)
        
        # clean-up
        del op
        del optimizer
        
        # write results to h5
        note = f"{'_ic' if ic else ''}{'_dc' if dc else ''}"
        h5_write = h5py.File(f'{output_dir_acc}/results.h5', 'w')
        h5_write.create_dataset(f'cgsense{note}', data=medutils.visualization.flip(cgsense, (1,2)))
        h5_write.create_dataset(f'tol{note}', data=acc)
        h5_write.create_dataset(f'img_regridded', data=medutils.visualization.flip(img_regridded))
        h5_write.close()

# generate reconstructions for heart
if 'rawdata_heart_radial_55proj_34ch.h5' in fname:
    print(f'intensity_compensation={ic}, density_compenation={dc}')
    alpha = 0.2
    for num_proj in [55, 33, 22, 11]:
        kspace_acc = kspace[...,:num_proj,:]
        nFE, nSpokes, nCH = kspace_acc.shape
        trajectory_acc = trajectory[...,:num_proj]

        output_dir_acc = fname.split('.h5')[0] + f'_proj{num_proj}'

        # prepare trajectory
        trajectory_col, dcf_col = ismrmrrsg.prepareTrajectory(trajectory_acc, save_dir=output_dir_acc, save_fig=True)
        kspace_radial = np.ascontiguousarray(np.reshape(np.transpose(kspace_acc, (2,0,1)), (nCh, nFE*nSpokes)))*np.sqrt(dcf_col)

        kspace_regridded = ismrmrrsg.regridRadialData(kspace_acc, trajectory_col, dcf_col, nufft_config, use_cg=use_cg_regrid)
        
        # compute regridded rss reconstruction
        img_regridded = medutils.mri.ifft2c(kspace_regridded, axes=(0,1))

        op = primaldualtoolbox.mri.MriRadialOperator(nufft_config)
        op.setTrajectory(trajectory_col)
        op.setDcf(dcf_col if dc else np.ones_like(dcf_col))
        op.setCoilSens(np.nan_to_num(smaps/icmap) if ic else smaps)
        optimizer = medutils.optimization.CgSenseReconstruction(op=op, max_iter=100, alpha=alpha, tol=1e-20)
        cgsense, acc = optimizer.solve(kspace_radial, return_tol=True, return_series=True)
        del op
        del optimizer
        note = f"{'_ic' if ic else ''}{'_dc' if dc else ''}"
        h5_write = h5py.File(f'{output_dir_acc}/results.h5', 'w')
        h5_write.create_dataset(f'cgsense{note}', data=medutils.visualization.flip(cgsense, (1,2)))
        h5_write.create_dataset(f'tol{note}', data=acc)
        h5_write.create_dataset(f'img_regridded', data=medutils.visualization.flip(img_regridded))
        h5_write.close()

# generate required plots
ismrmrrsg.plot(fname, ic, dc)
