import os
import yaml
import argparse
import numpy as np
# avoid warnings by turning off display
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import re

g_cmap = 'plasma'
g_figsize = (10, 10)
g_include_colorbar = False
g_include_title = False
g_use_grid = True
g_savefig_params = {'bbox_inches': 'tight', 'pad_inches': 0}

def visualize_marginal_single(marginal_pdfs, output_path, conf):
    discrete_num = conf['discrete_num']
    list_desc = conf['distribution_list']
    num_sources = len(list_desc)
    x0, x1, y0, y1 = conf['discrete_extent']
    X = np.linspace(x0, x1, discrete_num)
    Y = np.linspace(y0, y1, discrete_num)
    XX, YY = np.meshgrid(X, Y)
    output_prefix = os.path.splitext(output_path)[0]
    for r in range(num_sources):
        for c in range(2):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=g_figsize)
            ZZ = marginal_pdfs[r, c, :, :]
            cs = ax.contourf(XX, YY, ZZ, cmap=g_cmap)
            ax.set_aspect('equal')
            ax.set_xlim((x0, x1))
            ax.set_ylim((y0, y1))
            if c == 0:
                name = list_desc[r]['name']
            else:
                name = 'target'
            title = 'P_{}_(pi_{})_{}'.format(c, r, name)
            if g_use_grid:
                ax.grid(True)
            if g_include_colorbar:
                fig.colorbar(cs, ax=ax)
            ax.axis("off")
            fig.savefig('{}_{}'.format(output_prefix, title), **g_savefig_params)
            plt.close()

    # plot merged target marginals
    merged_target_pdf = np.mean(marginal_pdfs[:, 1, :, :], axis=0)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=g_figsize)
    ZZ = merged_target_pdf
    cs = ax.contourf(XX, YY, ZZ, cmap=g_cmap)
    ax.set_aspect('equal')
    ax.set_xlim((x0, x1))
    ax.set_ylim((y0, y1))
    title = 'target_merged'
    if g_use_grid:
        ax.grid(True)
    if g_include_colorbar:
        fig.colorbar(cs, ax=ax)
    ax.axis("off")
    fig.savefig('{}_{}'.format(output_prefix, title), **g_savefig_params)
    plt.close()

def visualize_potential_single(potential_vals, output_path, conf):
    discrete_num = conf['discrete_num']
    list_desc = conf['distribution_list']
    num_sources = len(list_desc)
    x0, x1, y0, y1 = conf['discrete_extent']
    X = np.linspace(x0, x1, discrete_num)
    Y = np.linspace(y0, y1, discrete_num)
    XX, YY = np.meshgrid(X, Y)
    fig, axes = plt.subplots(nrows=num_sources, ncols=2, figsize=g_figsize)
    for r, row in enumerate(axes):
        ax = row
        for c, ax in enumerate(row):
            ZZ = potential_vals[r, c, :, :]
            cs = ax.contourf(XX, YY, ZZ, cmap=g_cmap)
            ax.set_aspect('equal')
            ax.set_xlim((x0, x1))
            ax.set_ylim((y0, y1))
            if c == 0:
                title = 'f_{}_0'.format(r)
            else:
                title = 'f_{}_1'.format(r)
            if g_include_title:
                ax.set_title(title)
            if g_use_grid:
                ax.grid(True)
            if g_include_colorbar:
                fig.colorbar(cs, ax=ax)
            ax.axis("off")
    fig.savefig(output_path, **g_savefig_params)
    plt.close()

def visualize_sources(npy_dir, vis_dir, vis_name, conf):
    list_desc = conf['distribution_list']
    # preserve order
    sources = [s['name'] for s in list_desc]
    npys = [s + '.npy' for s in sources]
    num_sources = len(sources)

    discrete_num = conf['discrete_num']
    x0, x1, y0, y1 = conf['discrete_extent']
    X = np.linspace(x0, x1, discrete_num)
    Y = np.linspace(y0, y1, discrete_num)
    XX, YY = np.meshgrid(X, Y)
    fig, axes = plt.subplots(nrows=num_sources, ncols=1, figsize=g_figsize)
    for r, row in enumerate(axes):
        ax = row
        ZZ = np.load(os.path.join(npy_dir, npys[r]))
        cs = ax.contourf(XX, YY, ZZ, cmap=g_cmap)
        ax.set_aspect('equal')
        ax.set_xlim((x0, x1))
        ax.set_ylim((y0, y1))
        if g_include_title:
            ax.set_title(os.path.splitext(npys[r])[0])
        if g_use_grid:
            ax.grid(True)
        if g_include_colorbar:
            fig.colorbar(cs, ax=ax)
        ax.axis('off')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    output_path = os.path.join(vis_dir, vis_name)
    fig.savefig(output_path, **g_savefig_params)

    plt.close()

def visualize_pushforward_single(map_samples, output_path, conf):
    list_desc = conf['distribution_list']
    num_sources = len(list_desc)
    x0, x1, y0, y1 = conf['discrete_extent']
    fig, axes = plt.subplots(nrows=num_sources, ncols=2, figsize=g_figsize)
    for r, row in enumerate(axes):
        ax = row
        for c, ax in enumerate(row):
            ps = map_samples[r, c % 2, :, :]
            cs = ax.scatter(ps[:, 0], ps[:, 1], s=0.2, c='blue' if c % 2 == 0 else 'green')
            ax.set_aspect('equal')
            ax.set_xlim((x0, x1))
            ax.set_ylim((y0, y1))
            txt = 'from' if c % 2 == 0 else 'to'
            title = 'T_{}_{}'.format(r, txt)
            if g_include_title:
                ax.set_title(title)

            if g_use_grid:
                ax.grid(True)
            ax.axis("off")
    fig.savefig(output_path, **g_savefig_params)
    plt.close()

    # visualize merged samples
    merged_samples = np.concatenate([map_samples[i, 1, :, :] for i in range(num_sources)], axis=0)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=g_figsize)
    ax.scatter(merged_samples[:, 0], merged_samples[:, 1], s=0.5)
    ax.set_aspect('equal')
    ax.set_xlim((x0, x1))
    ax.set_ylim((y0, y1))
    if g_use_grid:
        ax.grid(True)
    ax.axis("off")
    fig.savefig(os.path.splitext(output_path)[0] + '_merged.png', **g_savefig_params)
    plt.close()

def visualize_mcmc_samples(all_samples, vis_dir, vis_name, conf, subsample=True):
    fig, axes = plt.subplots(nrows=all_samples.shape[0], ncols=2, figsize=g_figsize)
    x0, x1, y0, y1 = conf['discrete_extent']
    for r, row in enumerate(axes):
        ax = row
        for c, ax in enumerate(row):
            ps = all_samples[r, c, :, :]
            ax.scatter(ps[:, 0], ps[:, 1], s=0.2)
            ax.set_aspect('equal')
            ax.set_xlim((x0, x1))
            ax.set_ylim((y0, y1))
            if g_include_title:
                ax.set_title('P_{}#(pi_{})'.format(c, r))
            if g_use_grid:
                ax.grid(True)
            ax.axis('off')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    fig.savefig(os.path.join(vis_dir, vis_name), **g_savefig_params)
    plt.close()

    # Visualize merged samples.
    merged_samples = np.reshape(all_samples[:, 1, :, :], [-1, 2])
    if subsample:
        merged_inds = np.random.choice(merged_samples.shape[0], all_samples.shape[2])
        merged_samples = merged_samples[merged_inds, :]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=g_figsize)
    ax.scatter(merged_samples[:, 0], merged_samples[:, 1], s=0.5)
    ax.set_aspect('equal')
    ax.set_xlim((x0, x1))
    ax.set_ylim((y0, y1))
    if g_use_grid:
        ax.grid(True)
    ax.axis("off")
    fig.savefig(os.path.join(vis_dir, os.path.splitext(vis_name)[0] + '_merged.png'), **g_savefig_params)
    plt.close()


def visualize_barycenter_samples(npy_dir, npy_file, vis_dir, vis_name, conf):
    x0, x1, y0, y1 = conf['discrete_extent']
    ps = np.load(os.path.join(npy_dir, npy_file))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=g_figsize)
    assert(ps.shape[1] == 2) # can only visualize 2D
    ax.scatter(ps[:, 0], ps[:, 1], s=0.5)
    ax.set_aspect('equal')
    ax.set_xlim((x0, x1))
    ax.set_ylim((y0, y1))
    if g_include_title:
        ax.set_title(name)
    if g_use_grid:
        ax.grid(True)
    ax.axis("off")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    fig.savefig(os.path.join(vis_dir, vis_name), **g_savefig_params)
    plt.close()

# def sort_by_step(fs):
#     p = re.compile('.*-([0-9]+)\..*')
#     get_step = lambda f: int(p.match(f).group(1))
#     fs.sort(key=get_step)

def visualize_folder(npy_dir, vis_dir, vis_fn, conf):
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    npy_files = [f for f in os.listdir(npy_dir) if os.path.splitext(f)[1] == '.npy']
    # sort_by_step(npy_files)
    for f in npy_files:
        npy_path = os.path.join(npy_dir, f)
        npy_data = np.load(npy_path)
        output_path = os.path.join(vis_dir, os.path.splitext(f)[0] + '.png')
        vis_fn(npy_data, output_path, conf)

def create_vis_fn_from_name(name):
    if name == 'marginal':
        return visualize_marginal_single
    elif name == 'potential':
        return visualize_potential_single
    elif name == 'density':
        return visualize_density_single
    elif name == 'map' or name == 'potential_gradient' or name == 'barycentric_projection':
        return visualize_pushforward_single
    else:
        raise Exception('Unrecognized visualize name: {}'.format(name))
