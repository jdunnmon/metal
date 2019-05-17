import argparse
from functools import partial
import numpy as np
from os import listdir,mkdir, makedirs
from os.path import isdir,join,basename,normpath,split
import scipy.misc
from skimage import exposure, transform

from PIL import Image
from multiprocessing import Pool
from shutil import copyfile, rmtree

def f(ii, orig_path, save_path, list_imgs):
    """ function to map for multiprocessing
    """
    if ii>999 and ii % 1000 == 0:
        print('%0.4f Percent Complete' % (float(ii)/float(len(list_imgs))))
    pth = list_imgs[ii]
    old_pth = join(orig_path,pth)
    new_pth = join(save_path,pth)
    create_new_img(old_pth,new_pth)

def create_new_img(old_img_path,new_img_path):
    """
    Assumes each subfolder has an h5 and json containing same name
    """    
    old_img = Image.open(old_img_path,'r')
    eq = exposure.equalize_hist(np.array(old_img))
    scpimg = scipy.misc.toimage(eq,mode='P')
    scpimg.save(new_img_path,format='png')
    return 
        
def get_parser():
    parser = argparse.ArgumentParser(
            description="Train MetalModel on single or multiple tasks.", add_help=False
        )

    parser.add_argument(
        "--orig_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=12,
    )
    
    parser.add_argument(
        "--overwrite",
        type=int,
        default=1,
    )
    
    
    return parser

if __name__ == "__main__":
   
    parser = get_parser()
    args = parser.parse_args()
    
    if not isdir(args.save_dir):
        makedirs(args.save_dir)

    list_pats = listdir(args.orig_dir)
    existing = listdir(args.save_dir)
    list_pats = [a for a in list_pats if 'png' in a]
    
    if not args.overwrite:
        print('Not overwriting existing...')
        list_pats = list(set(list_pats) - set(existing))
    
    pl = Pool(args.n_threads)
    map_fun = partial(f,orig_path=args.orig_dir, save_path=args.save_dir, list_imgs=list_pats)
    pl.map(map_fun, range(len(list_pats)))

    print('Preprocessing complete!')