import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import numpy.ma as ma
keys = np.array(["0", "o", "O", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"])
low_let_ind = np.append(np.array([1]), np.arange(12, 12+25))
keys_low_letters = keys[low_let_ind]
values = np.array(list(range(len(keys))))
y_values = np.array([0, 0, 0] + list(range(1, 10)) + list(range(10, 10 + 25)) + list(range(10, 10 + 25)))
y_3dim = np.tile(y_values,(2, 11, 1)).T
y_low3dim = np.tile(low_let_ind, (2, 11, 1)).T
lexicon = dict(zip(keys, values))
anti_lexicon = dict(zip(values, keys))
frame_size = np.array([13.6, 20.4]) 
UJI_ratio, UPV_ratio = 100, 152
max_d = 392
N_wr = 11

def transform_set(dataset, frame=frame_size, ratio=UJI_ratio, angle=50):
    new_data = dataset.copy()
    div = frame[0]
    if frame[1] > frame[0]:
        div = frame[1]
    new_data /= div*ratio
    new_data -= new_data.mean(axis=-1)[..., np.newaxis]
    
    slant = new_data[..., :-1] - new_data[..., 1:]
    slant /= np.linalg.norm(slant, axis=-2)[..., np.newaxis, :]
    slant[..., np.tile((slant[..., 1, :] < 0)[..., np.newaxis, :], (1, 1, 1, 2, 1))] *= -1
    slant /= np.linalg.norm(slant, axis=-2)[..., np.newaxis, :]
    print()
    thres = np.cos(angle/180*np.pi)
    slant[..., np.tile((slant[...,1, :] < thres)[..., np.newaxis,:], (1, 1, 1, 2, 1))] = 0
    
    slant = slant.sum(axis=-1)
    slant /= np.linalg.norm(slant, axis=-1)[..., np.newaxis]
    
    sina = slant[..., 0]
    cosa = slant[..., 1]
    #new_data = np.dot()
    slant = slant[..., [[1, 0],[0, 1]]]
    slant[..., 0, 1] *= -1
    #slant[..., [0, 1], [1, 0]] *= np.sign(slant[..., 0])
    new_data = np.einsum('...ij,...jk->...ik', slant, new_data)
    return ma.masked_array(new_data, mask=dataset.mask)

def vis_letters(letter_index, real_data,frame=frame_size, ratio=UJI_ratio,
                vert_lines=np.array([7.5,12.7]), figsize=frame_size/2, verbose=False):
    points_xy = real_data[letter_index].compressed().reshape(2, -1)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    ax.plot(points_xy[0], points_xy[1], 'k-', linewidth=0.5)
    ax.scatter(points_xy[0, 1:-1], points_xy[1, 1:-1], c='b', marker='o', s=30)
    ax.scatter(points_xy[0, 0], points_xy[1, 0], c=(0, 0.95, 0), marker='o', s=40) #start
    ax.scatter(points_xy[0, -1], points_xy[1, -1], c=(0.95, 0, 0), marker='o', s=40) #end
    xlim = [0.0, frame[0]*ratio]
    ylim = [0.0, frame[1]*ratio]
    ax.xaxis.tick_top()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.plot(xlim, vert_lines[[0, 0]]*ratio, 'k', lw=1)
    ax.plot(ylim, vert_lines[[1, 1]]*ratio, 'k', lw=1)
    ax.set_xticks(np.arange(*(xlim + [100])))
    ax.set_yticks(np.arange(*(ylim + [100])))
    # We change the fontsize of minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=figsize[0]*2)
    if verbose:
        ax.set_title("Буква '{0:s}'\tреспондент {1:d}\tпопытка {2:d}".format(
                anti_lexicon[letter_index[0]], letter_index[1] + 1, letter_index[2] + 1),fontsize=figsize[1]*2, y=1.03)
    ax.invert_yaxis()
    ax.grid()

def vis_letters_trans(letter_index, real_data,frame=frame_size, ratio=UJI_ratio,
                 vert_lines=np.array([7.5,12.7]), figsize=frame_size/2,verbose=False):
    points_xy = real_data[letter_index].compressed().reshape(2, -1)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    ax.plot(points_xy[0], points_xy[1], 'k-', linewidth=0.5)
    ax.scatter(points_xy[0, 1:-1], points_xy[1, 1:-1], c='b', marker='o', s=30)
    ax.scatter(points_xy[0, 0], points_xy[1, 0], c=(0, 0.95, 0), marker='o', s=40) #start
    ax.scatter(points_xy[0, -1], points_xy[1, -1], c=(0.95, 0, 0), marker='o', s=40) #end
    
    ylim = [-0.5, 0.5]
    xlim = [-0.5, 0.5]
    ax.xaxis.tick_top()
    ax.set_aspect('equal')
    #ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    #ax.plot(xlim, vert_lines[[0, 0]]*ratio, 'k', lw=1)
    #ax.plot(ylim, vert_lines[[1, 1]]*ratio, 'k', lw=1)
    ax.set_xticks(np.arange(*xlim, step=0.25))
    ax.set_yticks(np.arange(*ylim, step=0.25))
    # We change the fontsize of minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=figsize[0]*2)
    if verbose:
        ax.set_title("Буква '{0:s}'\tреспондент {1:d}\tпопытка {2:d}".format(
                anti_lexicon[letter_index[0]], letter_index[1] + 1, letter_index[2] + 1),
                    fontsize=figsize[1]*2, y=1.03)
    ax.invert_yaxis()
    ax.grid()

def dist_vis(Dist, name='colorMap', extra_values=keys):
    fig = plt.figure(figsize=(15, 8))
    if extra_values is not None:
        sym_num = extra_values.size
    ax = fig.add_subplot(111)
    ax.set_title(name)
    plt.imshow(Dist, interpolation='none')
    ax.set_aspect('equal')
    if extra_values is not None:
        ax.set_yticks(np.arange(sym_num))
        ax.set_yticklabels(list(extra_values), rotation=90, fontsize=10)
        ax.set_xticks(np.arange(sym_num))
        ax.set_xticklabels(list(extra_values), rotation=0, fontsize=10)
    ax.xaxis.grid(), ax.yaxis.grid()
    
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()

#new_real_data = transform_set(real_data)
#new2_real_data = ma.empty_like(new_real_data)
#with change_printopt(threshold=1000):
#    for letter in range(real_data.shape[0]):
#        for wr in range(real_data.shape[1]):
#            for rep in range(2):
#                new2_real_data[letter, wr, rep] = transform(real_data[letter, wr, rep])
#                if not ma.allclose(new2_real_data[letter, wr, rep],
#                                   new_real_data[letter, wr, rep]) and letter < 3:
#                    print(letter, wr, rep, sep=', ',flush=1)
#                    print(np.max(np.abs(new2_real_data[letter, wr, rep] -
#                        new_real_data[letter, wr, rep]).compressed()))

#def vis_letters2(letter_index, real_data,frame=frame_size, ratio=UJI_ratio,
#                 vert_lines=np.array([7.5,12.7]), figsize=frame_size/2, verbose=True):
#    points_xy = transform(real_data[letter_index]).compressed().reshape(2, -1)
#    fig = plt.figure(figsize=figsize)
#    ax = fig.gca()
#    ax.plot(points_xy[0], points_xy[1], 'k-', linewidth=0.5)
#    ax.scatter(points_xy[0, 1:-1], points_xy[1, 1:-1], c='b', marker='o', s=30)
#    ax.scatter(points_xy[0, 0], points_xy[1, 0], c=(0, 0.95, 0), marker='o', s=40) #start
#    ax.scatter(points_xy[0, -1], points_xy[1, -1], c=(0.95, 0, 0), marker='o', s=40) #end
    
#    ylim = [-0.5, 0.5]
#    xlim = [-0.5, 0.5]
#    ax.xaxis.tick_top()
#    ax.set_aspect('equal')
#    #ax.set_xlim(*xlim)
#    ax.set_ylim(*ylim)
#    #ax.plot(xlim, vert_lines[[0, 0]]*ratio, 'k', lw=1)
#    #ax.plot(ylim, vert_lines[[1, 1]]*ratio, 'k', lw=1)
#    ax.set_xticks(np.arange(*xlim, step=0.25))
#    ax.set_yticks(np.arange(*ylim, step=0.25))
#    # We change the fontsize of minor ticks label 
#    ax.tick_params(axis='both', which='major', labelsize=figsize[0]*2)
#    if verbose:
#        ax.set_title("Буква '{0:s}'\tреспондент {1:d}\tпопытка {2:d}".format(
#                anti_lexicon[letter_index[0]], letter_index[1] + 1, letter_index[2] + 1),fontsize=figsize[1]*2, y=1.03)
#    ax.invert_yaxis()
#    ax.grid()     