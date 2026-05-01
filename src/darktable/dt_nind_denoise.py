#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: Huy Hoang

Darktable NIND Denoise Pipeline.

Usage:
  dt_nind_denoise.py [options] <filenames>...
  dt_nind_denoise.py -h | --help

Options:
  -h --help             Show this screen.
  -n --nightmode        Enable nightmode (normalize brightness before denoise).
  -r --rating=RATING    Darktable rating filter [default: 012345].
  -d --debug            Enable debug mode (verbose output).
  --no-rldeblur         Disable RL-deblur (GMIC). It is ON by default.
  -o --outdir=DIR       Output directory [default: darktable_exported].
  -e --ext=EXT          Output file extension [default: jpg].
  -q --quality=QUAL     JPEG compression quality [default: 90].
  -s --sigma=SIGMA      RL-deblur sigma [default: 1.0].
  -i --iter=ITER        RL-deblur number of iterations [default: 10].
"""

import sys
import subprocess
import shutil
import logging
import configparser
from typing import Dict, Any, List
from pathlib import Path

from docopt import docopt
from bs4 import BeautifulSoup
import exiv2

# --- ANSI Color Constants ---
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

VALID_EXTENSIONS = {'.3FR', '.ARW', '.SR2', '.SRF', '.CR2', '.CR3', '.CRW',
                    '.DNG', '.ERF', '.FFF', '.MRW', '.NEF', '.NRW', '.ORF',
                    '.PEF', '.RAF', '.RW2'}

FIRST_OPS = [
    'channelmixerrgb', 'colorin', 'colorout', 'demosaic', 'flip',
    'gamma', 'highlights', 'hotpixels', 'mask_manager', 'rawprepare', 'temperature'
]

SECOND_OPS = [
    'ashift', 'bilat', 'blurs', 'borders', 'colorbalancergb', 'colorin',
    'colorout', 'crop', 'cacorrectrgb', 'clahe', 'denoiseprofile',
    'diffuse', 'dither', 'exposure', 'filmicrgb', 'flip', 'gamma',
    'hazeremoval', 'invert', 'lens', 'levels', 'liquify', 'lowlight',
    'lut3d', 'mask_manager', 'monochrome', 'nlmeans', 'rgbcurve',
    'rgblevels', 'rotatepixels', 'scalepixels', 'shadhi', 'sharpen',
    'soften', 'splittoning', 'spots', 'tonecurve', 'tonemap',
    'toneequal', 'velvia', 'vibrance', 'vignette', 'watermark', 'zonesystem'
]

SECOND_OVERRIDES = {
    'colorin': {
        'darktable:num': "0",
        'darktable:operation': "colorin",
        'darktable:enabled': "1",
        'darktable:modversion': "7",
        'darktable:params': "gz48eJxjZBgFowABWAbaAaNgwAEAEDgABg==",
        'darktable:blendop_version': "14",
        'darktable:blendop_params': "gz11eJxjYIAACQYYOOHEgAZY0QWAgBGLGANDgz0Ej1Q+dcF/IADRAGpyHQU="
    }
}

class DenoisePipeline:
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.config = self._load_config()
        self.out_dir = Path(args['--outdir'])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        log_level = logging.DEBUG if args['--debug'] else logging.INFO
        logging.basicConfig(level=log_level, format='%(message)s')
        self.logger = logging.getLogger(__name__)

    def _load_config(self) -> configparser.ConfigParser:
        config_path = Path(__file__).parent / 'dt_nind_denoise.ini'
        if not config_path.exists():
            print(f"Error: {config_path} not found.")
            sys.exit(1)
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def _get_unique_path(self, filename: str) -> Path:
        """Finds a non-conflicting filename in the output directory."""
        base = self.out_dir / filename
        if not base.exists():
            return base

        counter = 1
        while True:
            new_path = self.out_dir / f"{base.stem}_{counter:02d}{base.suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    def modify_xmp(self, xmp_path: Path, stage: int) -> Path:
        with open(xmp_path, 'r') as f:
            soup = BeautifulSoup(f.read(), "xml")

        history = soup.find('darktable:history')
        if not history:
            return xmp_path

        ops = history.find_all('rdf:li')
        ops.sort(key=lambda x: int(x['darktable:num']))

        current_first = list(FIRST_OPS)
        current_second = list(SECOND_OPS)
        if self.args['--nightmode']:
            for op in ['exposure', 'toneequal']:
                current_first.append(op)
                if op in current_second: current_second.remove(op)

        self.logger.debug(f"\n--- Stage {stage} History Modification ---")
        for op in reversed(ops):
            name = op['darktable:operation']

            if stage == 1:
                if name not in current_first:
                    self.logger.debug(f"Stage 1: {RED}[REMOVED]{RESET} {name}")
                    op.extract()
                else:
                    self.logger.debug(f"Stage 1: {GREEN}[KEEP]{RESET}    {name}")
                    if name == 'flip':
                        op['darktable:enabled'] = "0"

            else: # Stage 2
                if name not in current_second and name in current_first:
                    self.logger.debug(f"Stage 2: {RED}[REMOVED]{RESET} {name}")
                    op.extract()
                else:
                    self.logger.debug(f"Stage 2: {GREEN}[KEEP]{RESET}    {name}")
                    if name in SECOND_OVERRIDES:
                        for key, val in SECOND_OVERRIDES[name].items():
                            op[key] = val

        if stage == 2:
            desc = soup.find('rdf:Description')
            desc['darktable:iop_order_version'] = '5'
            if desc.has_attr("darktable:iop_order_list"):
                val = desc['darktable:iop_order_list']
                desc['darktable:iop_order_list'] = val.replace('colorin,0,', '').replace('demosaic,0', 'demosaic,0,colorin,0')

        output_xmp = xmp_path.with_suffix(f'.s{stage}.xmp')
        with open(output_xmp, 'w') as f:
            f.write(soup.prettify())
        return output_xmp

    def run_cmd(self, cmd: str):
        self.logger.debug(f"Executing: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    def process_image(self, img_path: str):
        path = Path(img_path)
        if not path.exists() or path.suffix.upper() not in VALID_EXTENSIONS:
            self.logger.warning(f"Skipping invalid/non-RAW: {img_path}")
            return

        xmp_path = Path(f"{img_path}.xmp")
        self.logger.info(f"\nProcessing: {path.name}")

        # Working files
        s1_tif = path.with_name(f"{path.stem}_s1.tif")
        denoised_tiff = path.with_name(f"{path.stem}_s1_denoised.tiff")
        s2_tif = path.with_name(f"{path.stem}_s2.tif")

        # Output target
        target_ext = f".{self.args['--ext'].lstrip('.')}"
        final_out = self._get_unique_path(f"{path.stem}{target_ext}")

        try:
            # Orphan cleanup
            for f in [s1_tif, denoised_tiff, s2_tif]:
                f.unlink(missing_ok=True)

            # 1. Stage 1 Export
            s1_xmp = self.modify_xmp(xmp_path, stage=1)
            dt_bin = self.config['command']['darktable'].strip()
            self.run_cmd(f'{dt_bin} "{path}" "{s1_xmp}" "{s1_tif}" --apply-custom-presets 0 --core --conf plugins/imageio/format/tiff/bpp=32')

            # 2. NIND Denoise
            nind_bin = self.config['command']['nind_denoise']
            nind_params = self.config['command']['nind_denoise_params']
            self.run_cmd(f'{nind_bin} {nind_params} --input "{s1_tif}" --output "{denoised_tiff}"')

            # 3. Metadata Transfer
            self._copy_meta(path, denoised_tiff)

            # 4. Stage 2 Export
            s2_xmp = self.modify_xmp(xmp_path, stage=2)
            stage2_target = s2_tif if not self.args['--no-rldeblur'] else final_out

            self.run_cmd(f'{dt_bin} "{denoised_tiff}" "{s2_xmp}" "{stage2_target}" --icc-intent PERCEPTUAL --icc-type SRGB --apply-custom-presets 0 --core --conf plugins/imageio/format/tiff/bpp=16')

            # 5. RL-Deblur
            if not self.args['--no-rldeblur'] and 'gmic' in self.config['command']:
                self.logger.info("Applying RL-Deblur...")
                gmic_bin = self.config['command']['gmic'].strip()
                tmp_rl = final_out.parent / f"tmp_deblur_{final_out.name.replace(' ', '_')}"

                gmic_cmd = (f'{gmic_bin} "{s2_tif}" -deblur_richardsonlucy {self.args["--sigma"]},{self.args["--iter"]},1 '
                            f'-/ 256 cut 0,255 round -o "{tmp_rl},{self.args["--quality"]}"')
                self.run_cmd(gmic_cmd)
                shutil.move(str(tmp_rl), str(final_out))
                self._copy_meta(s1_tif, final_out)

            self.logger.info(f"Successfully exported to: {final_out}")

        finally:
            if not self.args['--debug']:
                self._cleanup([s1_tif, denoised_tiff, s1_xmp, s2_xmp, s2_tif])

    def _copy_meta(self, src: Path, dst: Path):
        try:
            s = exiv2.ImageFactory.open(str(src))
            s.readMetadata()
            d = exiv2.ImageFactory.open(str(dst))
            d.setExifData(s.exifData())
            d.writeMetadata()
        except Exception as e:
            self.logger.error(f"Metadata transfer failed for {dst.name}: {e}")

    def _cleanup(self, files: List[Path]):
        for f in files:
            if f and f.exists(): f.unlink()

def main():
    args = docopt(__doc__)

    # Cast docopt parameters
    args['--quality'] = int(args['--quality'])
    args['--sigma'] = float(args['--sigma'])
    args['--iter'] = int(args['--iter'])

    pipeline = DenoisePipeline(args)
    for f in args['<filenames>']:
        pipeline.process_image(f)

if __name__ == "__main__":
    main()