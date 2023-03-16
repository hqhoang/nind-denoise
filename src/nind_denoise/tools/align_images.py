#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# License : GPLv3 : http://gplv3.fsf.org/


from PIL import Image
from cv2 import cv2
import numpy as np
import multiprocessing as mp
import timeit
import os, sys
from argparse import ArgumentParser
import math




class OpenCV_Aligner():
  iteration = 20
  ter_eps = 1e-1
  pool_size = math.floor(mp.cpu_count()/2)
  jpg_quality = 90


  """
    image_list: [ [input_filename, output_filename] ]
    options:
      'iteration'   : number of iterations
      'ter_eps'     : termination epsilon
      'pool_size'   : pool size (how many threads to run in parallel)
      'jpg_quality' : JPG compression quality

  """
  def align(self, image_list, options = {}):
    """ root-level function for multiprocessing """
    global pool

    if 'iteration' in options:
      self.iteration = options['iteration']

    if 'ter_eps' in options:
      self.ter_eps = options['ter_eps']

    if 'jpg_quality' in options:
      self.jpg_quality = options['jpg_quality']

    if 'pool_size' in options and options['pool_size'] > 0:
      self.pool_size = options['pool_size']
      self.pool_size = min(self.pool_size, mp.cpu_count())

    if 'anchor_index' in options:
      anchor_index = options['anchor_index']
    if anchor_index < 0:
      anchor_index = math.floor(len(image_list)/2)


    # write out anchor image as-is
    anchor_img = Image.open(image_list[anchor_index][0])
    anchor_img.save(image_list[anchor_index][1], quality=self.jpg_quality)

    print('\nUsing "' + os.path.basename(image_list[anchor_index][0]) + '" as anchor, ' +\
          'written to ' + image_list[anchor_index][1])

    # initiate a pool
    pool = mp.Pool(self.pool_size)
    print('\nInitated a pool of ' + str(self.pool_size) + ' workers, termination epsilon = ' +\
          str(options['ter_eps']) + ', ' + str(options['iteration']) + ' iterations\n')

    worker_options = {
      'iteration'   :  int(self.iteration),
      'ter_eps'     :  float(self.ter_eps),
      'jpg_quality' :  int(self.jpg_quality)
    }

    results = []
    for i, filepaths in enumerate(image_list):
      if i != anchor_index:
        # important: do not pass any widget to apply_async since we're copying the parent into the child processes
        result = pool.apply_async(self.align_pyramid, (str(image_list[anchor_index][0]), str(filepaths[0]), str(filepaths[1]), worker_options.copy()))

        # for single-process debugging:
        # result = self.align_pyramid(str(image_list[anchor_index][0]), str(filepaths[0]), str(filepaths[1]), worker_options.copy())
        results.append(result)

    for result in results:
      if result:
        result.get()    # needed to catch any error/exception from subprocesses


    # close Pool and let all the processes complete
    pool.close()
    pool.join()  # wait for all processes



  """
    pyramid algorithm from https://stackoverflow.com/questions/45997891/cv2-motion-euclidean-for-the-warp-mode-in-ecc-image-alignment-method
  """
  def align_pyramid(self, anchor_filepath, target_filepath, out_filepath, options):
    # global main_queue

    print('ECC aligning ' + os.path.basename(target_filepath) + ' against ' + os.path.basename(anchor_filepath))

    anchor_img = cv2.imread(anchor_filepath)
    anchor_img_gray = cv2.cvtColor(anchor_img, cv2.COLOR_RGB2GRAY)

    iteration = options['iteration']
    ter_eps   = options['ter_eps']

    pyramid_level = None
    if 'pyramid_level' in options:
      pyramid_level = options['pyramid_level']

    warp_mode = cv2.MOTION_HOMOGRAPHY

    # Initialize the matrix to identity
    warp_matrix = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)

    w = anchor_img_gray.shape[1]

    # determine number of levels
    if pyramid_level is None:
      nol =  math.floor((math.log(w/300, 2)))
    else:
      nol = pyramid_level

    # print('Number of levels: ' + str(nol))

    warp_matrix[0][2] /= (2**nol)
    warp_matrix[1][2] /= (2**nol)

    target_img = cv2.imread(target_filepath)
    target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)

    # construct grayscale pyramid
    gray1_pyr = [anchor_img_gray]
    gray2_pyr = [target_img_gray]

    # print('target_img: ', target_img_gray.shape)

    for level in range(nol):
      # print('level: ', level, ', gray1_pyr[0].shape: ', gray1_pyr[0].shape)

      gray1_pyr.insert(0, cv2.resize(gray1_pyr[0], None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA))
      gray2_pyr.insert(0, cv2.resize(gray2_pyr[0], None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA))

    # Terminate the optimizer if either the max iterations or the threshold are reached
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iteration, ter_eps )

    # run pyramid ECC
    pyr_start_time = timeit.default_timer()

    for level in range(nol+1):
      # lvl_start_time = timeit.default_timer()

      grad1 = gray1_pyr[level]
      grad2 = gray2_pyr[level]

      # print('level:', level, ', gray1_pyr[level].shape:', gray1_pyr[level].shape)

      cc, warp_matrix = cv2.findTransformECC(grad1, grad2, warp_matrix, warp_mode, criteria)

      if level < nol:
        # scale up for the next pyramid level
        warp_matrix = warp_matrix * np.array([[1,1,2],[1,1,2],[0.5,0.5,1]], dtype=np.float32)

      # print('Level %i time:'%level, timeit.default_timer() - lvl_start_time)

    # print('Pyramid time (', os.path.basename(target_filepath), '): ', timeit.default_timer() - pyr_start_time)

    # Get the target size from the desired image
    target_shape = anchor_img.shape

    aligned_img = cv2.warpPerspective(
                        target_img,
                        warp_matrix,
                        (target_shape[1], target_shape[0]),
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                        flags=cv2.INTER_AREA + cv2.WARP_INVERSE_MAP)

    #cv2.imwrite(out_filepath, aligned_img, [cv2.IMWRITE_JPEG_QUALITY, options['jpg_quality']])

    # copy EXIF from input to output
    input_image = Image.open(target_filepath)
    exif = input_image.info['exif']

    output_image = Image.fromarray(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
    output_image.save(out_filepath, quality=options['jpg_quality'], exif=exif)

    print('\twritten to: ' + out_filepath +\
          '\t (' + '{:.2f}'.format(timeit.default_timer() - pyr_start_time) + ' seconds)')




"""==============================================
  main program
=============================================="""
def main(argv):
  parser = ArgumentParser()
  parser.add_argument("filenames", metavar="FILE", nargs='*',
                      help="source image", )
  parser.add_argument("-e", "--extension", dest="extension", default='jpg',
                      help="output file extension, default: jpg")
  parser.add_argument("-q", "--quality", dest="quality", type=int, default=90,
                      help="JPEG compression quality, default: 90")
  parser.add_argument("-s", "--eps", dest="eps", type=float, default=1e-2,
                      help="ECC's termination epsilon, default: 1e-2")
  parser.add_argument("-i", "--iter", dest="iteration", type=int, default=50,
                      help="number of iteration, default: 50")
  parser.add_argument("-a", "--anchor-index", dest="anchor", type=int, default=-1,
                      help="index of the image to use as anchor (starts at 0), default: middle image")
  parser.add_argument("-t", "--threads", dest="threads", type=int, default=0,
                      help="number of threads to run in parallel, default: half of CPU count")

  args = parser.parse_args()

  # create output folder if needed
  outdir = 'aligned'
  if not os.path.exists(outdir):
      os.makedirs(outdir)

  # check input images and prepare output filename mapping
  images = []

  for filename in args.filenames:
    if not os.path.exists(filename):
      print('\nError: file ' + filename + ' does not exist.')
      return(1)

    # determine output filename
    basename, ext = os.path.splitext(filename)
    if args.extension != '':
        ext = '.' + args.extension.lstrip('.')
    out_filename = outdir + '/' + basename + ext

    images.append([filename, out_filename])


  # align the images
  opencv_aligner = OpenCV_Aligner()
  ecc_options = {
    'iteration'    : args.iteration,
    'ter_eps'      : args.eps,
    'pool_size'    : args.threads,
    'anchor_index' : args.anchor,
    'jpg_quality'  : args.quality
  }

  if (args.anchor >= len(images)):
    print('Anchor index is greater than the number of images, using middle image as anchor instead')
    ecc_options['anchor_index'] = -1

  opencv_aligner.align(images, ecc_options)
  print()


# =================
# start the program
if __name__ == "__main__":
  main_queue = mp.Queue()
  mp.freeze_support()

  main(sys.argv[1:])
  # global placeholder to work around multiprocessing
  pool = None