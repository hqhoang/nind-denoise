#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Huy Hoang

Split the darktable export into two stages to inject nind-denoise into the
history stack

"""

import os, sys, subprocess, argparse
from argparse import ArgumentParser
import configparser
from bs4 import BeautifulSoup
import copy

notouch_ops = [
]

# list of operations to be moved to second stage
post_ops = [
  'ashift',         # rotate & perspective, NOTE: autocrop doesn't auto-apply via non-interactive mode
  'bilat',          # local contrast
  'blurs',
  'borders',        # framing
  'colorbalancergb',
  'crop',
  'cacorrectrgb',   # chromatic aberrations
  'clahe',          # local contrast
  'denoiseprofile',
  'diffuse',        # diffuse or sharpen
  'dither',         # dithering
  'hazeremoval',
  'invert',
  'lens',           # lens correction
  'levels',
  'liquify',
  'lowlight',       # lowlight vision
  'lut3d',
  'monochrome',
  'nlmeans',        # astro photo denoise
  'rawdenoise',
  'rgbcurve',
  'rgblevels',
  'rotatepixels',
  'scalepixels',
  'shadhi',         # shadow and highlight
  'sharpen',
  'soften',
  'splittoning',
  'spots',          # spot removal
  'tonecurve',
  'tonemap',        # tone-mapping
  'velvia',
  'vibrance',
  'vignette',
  'watermark',
  'zonesystem',

  # the below should belong to 1st stage
  # 'exposure',
  # 'filmicrgb',     # it needs full data from original file
  # 'flip',          # orientation
  # 'gamma',
  # 'retouch',
  # 'sigmoid',       # it needs full data from original file
  # 'toneequal',     # tone-equalizer seems to depend on exposure to align the histogram
]



"""
  main program, meant to be called manually or by darktable's lua script

"""

def main(argv):
    parser = ArgumentParser()
    parser.add_argument("filenames", metavar="FILE", nargs='*',
                        help="source image", )
    parser.add_argument("-r", "--rating", dest="rating", default='012345',
                        help="darktable rating, specified as [012345], default: 012345 (all)")
    parser.add_argument("-d", "--debug", dest="debug", type=bool, action=argparse.BooleanOptionalAction,
                        help="debug mode to print extra info and keep intermedia files, default: no")
    parser.add_argument("-l", "--rldeblur", dest="rldeblur", type=bool, action=argparse.BooleanOptionalAction,
                        default=True, help="whether to enable RL-deblur, default: yes")
    parser.add_argument("-e", "--extension", dest="extension", default='jpg',
                        help="output file extension, default: jpg")
    parser.add_argument("-q", "--quality", dest="quality", type=int, default=90,
                        help="JPEG compression quality, default: 90")
    parser.add_argument("-s", "--sigma", dest="sigma", type=float, default=1,
                        help="RL-deblur sigma, default: 1")
    parser.add_argument("-i", "--iter", dest="iteration", type=int, default=10,
                        help="RL-deblur number of iteration, default: 10")

    args = parser.parse_args()

    # create output folder if needed
    outdir = 'darktable_exported'
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    # read config
    config_filename = os.path.dirname(__file__) + '/' + 'dt_nind_denoise.ini'

    if not os.path.exists(config_filename):
      print('Error reading ', config_filename)
      exit(1)

    config = configparser.ConfigParser()
    config.read(config_filename)

    cmd_darktable     = config['command']['darktable']
    cmd_nind_denoise  = config['command']['nind_denoise']
    cmd_exiftool      = config['command']['exiftool']
    cmd_gmic          = config['command']['gmic']


    # main loop: iterate through all provided images
    for filename in args.filenames:
      print("\n")

      # determine a new filename
      basename, ext = os.path.splitext(filename)

      if args.extension != '':
          ext = '.' + args.extension.lstrip('.')

      i = 0
      out_filename = outdir + '/' + basename + ext
      while (os.path.exists(out_filename)):
          i = i + 1
          out_filename = outdir + '/' + basename + '_' + str(i) + ext

      # read the XMP
      xmp = filename + '.xmp'

      if not os.path.exists(xmp):
        print("Error: cannot find sidecar file ", xmp)
        continue

      with open(xmp, 'r') as f:
          sidecar_xml = f.read()

      sidecar = BeautifulSoup(sidecar_xml, "xml")

      # check rating
      rating = sidecar.find('rdf:Description')['xmp:Rating']

      if args.debug:
        print('Rating: ', rating)

      rating_filter = list(args.rating)
      if rating not in rating_filter:
        print('Rating of', rating, 'does not match rating filter. Skipping.')
        continue

      # read the history stack
      history = sidecar.find('darktable:history')
      history_org = copy.copy(history)
      history_ops = history.find_all('rdf:li')


      # disable the post-ops then save to first stage
      if args.debug:
        print("\nPrepping first stage ...")

      for op in reversed(history_ops):
        if op['darktable:operation'] in post_ops:
          op['darktable:enabled'] = "0"

          if args.debug:
            print("--disabled: ", op['darktable:operation'])
        elif args.debug:
          print("default:    ", op['darktable:operation'])

      with open(filename+'.s1.xmp', 'w') as first_stage:
        first_stage.write(sidecar.prettify())


      # restore the history stack to original
      history.replace_with(history_org)
      history_ops = history_org.find_all('rdf:li')

      # enable only post-ops then save to second stage
      if args.debug:
        print("\nPrepping second stage ...")

      for op in reversed(history_ops):
        if op['darktable:operation'] in post_ops:
          if args.debug:
            print("default:    ", op['darktable:operation'], op['darktable:enabled'])
        else:
          if op['darktable:operation'] not in notouch_ops:
            op.extract()    # remove the op completely

            if args.debug:
              print("--removed: ", op['darktable:operation'])

      with open(filename+'.s2.xmp', 'w') as second_stage:
          second_stage.write(sidecar.prettify())


      # invoke darktable-cli with first stage
      s1_filename = outdir + '/' + basename + '_s1.tif'

      if os.path.exists(s1_filename):
        os.remove(s1_filename)

      cmd = cmd_darktable + ' "' + filename + '" "' + filename + '.s1.xmp" "' + s1_filename + '" ' + \
            '--apply-custom-presets 0 --core --conf plugins/imageio/format/tiff/bpp=32 '

      if args.debug:
        print('First-stage cmd: ', cmd)

      subprocess.call(cmd, shell=True)

      if not os.path.exists(s1_filename):
        print("Error: first-stage export not found: ", s1_filename)
        continue


      # call nind-denoise
      denoised_filename = outdir + '/' + basename + '_s1_denoised.tiff' # for nind-denoise: tif = 16-bit, tiff = 32-bit

      if os.path.exists(denoised_filename):
        os.remove(denoised_filename)

      cmd = cmd_nind_denoise + ' --input "' + s1_filename + '" --output "' + denoised_filename + '"'

      if args.debug:
        print('nind-denoise cmd: ', cmd)

      subprocess.call(cmd, shell=True)

      if not os.path.exists(s1_filename):
        print("Error: denoised image not found: ", denoised_filename)
        continue


      # copy exif from RAW file to denoised image
      cmd = cmd_exiftool + ' -writeMode cg -TagsFromFile "' + filename + '" -all:all -overwrite_original "' + denoised_filename + '"'
      if args.debug:
        print("exiftool cmd: ", cmd)

      subprocess.call(cmd, shell=True)
      print('Copied EXIF from ' + filename + ' to ' + denoised_filename)


      # invoke darktable-cli with second stage
      if args.rldeblur:
        s2_filename = outdir + '/' + basename + '_s2.tif'

        if os.path.exists(s2_filename):
          os.remove(s2_filename)
      else:
        s2_filename = out_filename

      cmd = cmd_darktable + ' "' + denoised_filename + '" "' + filename + '.s2.xmp" "' + s2_filename + '" ' + \
            '--apply-custom-presets 0 --core --conf plugins/imageio/format/tiff/bpp=16 '

      if args.debug:
        print('Second-stage cmd: ', cmd)

      subprocess.call(cmd, shell=True)


      # call ImageMagick RL-deblur
      if args.rldeblur:
        tmp_rl_filename = out_filename.replace(' ', '_')

        cmd = cmd_gmic + ' "' + s2_filename + '" -deblur_richardsonlucy ' + str(args.sigma) + ',' + str(args.iteration) + ',1' + \
              '  -/ 256 cut 0,255 round -o "' + tmp_rl_filename + ',' + str(args.quality) + '"'

        if args.debug:
          print('RL-deblur cmd: ', cmd)

        subprocess.call(cmd, shell=True)

        # rename tmp file
        os.rename(tmp_rl_filename, out_filename)
        print('Applied RL-deblur to:', out_filename)


      # copy exif
      cmd = cmd_exiftool + ' -writeMode cg -TagsFromFile "' + s2_filename + '" -all:all -overwrite_original "' + out_filename + '"'

      if args.debug:
        print("exiftool cmd: ", cmd)

      subprocess.call(cmd, shell=True)
      print('Copied EXIF to:', out_filename)


      # clean up
      if not args.debug:
        os.remove(s1_filename)
        os.remove(denoised_filename)
        os.remove(filename + '.s1.xmp')
        os.remove(filename + '.s2.xmp')

        if (s2_filename != out_filename and os.path.exists(s2_filename)):
          os.remove(s2_filename)



# =================
# start the program
if __name__ == "__main__":
   main(sys.argv[1:])