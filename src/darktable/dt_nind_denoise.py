#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: Huy Hoang

Darktable NIND Denoise & Natural Grain Pipeline.

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
        base = self.out_dir / filename
        if not base.exists():
            return base

        counter = 1
        while True:
            new_path = self.out_dir / f"{base.stem}_{counter:02d}{base.suffix}"
            if not new_path.exists():
                return new_path
            counter += 1


    def _get_rating(self, xmp_path: Path) -> str:
        """Extracts the xmp:Rating from the sidecar. Defaults to '0' if missing."""
        if not xmp_path.exists():
            return "0"
        try:
            with open(xmp_path, 'r') as f:
                soup = BeautifulSoup(f.read(), "xml")
            desc = soup.find('rdf:Description')
            if desc and desc.has_attr('xmp:Rating'):
                return desc['xmp:Rating']
        except Exception as e:
            self.logger.warning(f"Could not read rating from {xmp_path}: {e}")
        return "0"


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

            else:
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


    def create_grainy_xmp(self, original_xmp: Path, style_path: Path) -> Path:
        """
        Parses a Darktable .dtstyle file, extracts all operations, and injects
        them into the target XMP history stack.
        """
        if not style_path.exists():
            self.logger.error(f"Grain style not found: {style_path}")
            raise FileNotFoundError(f"Missing {style_path}")

        with open(original_xmp, 'r') as f:
            orig_soup = BeautifulSoup(f.read(), "xml")

        with open(style_path, 'r') as f:
            style_soup = BeautifulSoup(f.read(), "xml")

        plugins = style_soup.find_all('plugin')
        if not plugins:
            self.logger.warning(f"No operations found in dtstyle: {style_path}")
            return original_xmp

        history = orig_soup.find('darktable:history')
        if not history:
            return original_xmp

        seq = history.find('rdf:Seq') or history
        desc = orig_soup.find('rdf:Description')

        # Find the highest existing num in the sequence
        nums = [int(li.get('darktable:num', -1)) for li in seq.find_all('rdf:li')]
        highest_num = max(nums) if nums else -1

        for plugin in plugins:
            # Map <dtstyle> tags to <darktable:xxx> attributes
            operation = plugin.find('operation').text if plugin.find('operation') else ''
            modversion = plugin.find('module').text if plugin.find('module') else ''
            params = plugin.find('op_params').text if plugin.find('op_params') else ''
            enabled = plugin.find('enabled').text if plugin.find('enabled') else '1'
            blendop_params = plugin.find('blendop_params').text if plugin.find('blendop_params') else ''
            blendop_version = plugin.find('blendop_version').text if plugin.find('blendop_version') else ''
            multi_priority = plugin.find('multi_priority').text if plugin.find('multi_priority') else '0'
            multi_name = plugin.find('multi_name').text if plugin.find('multi_name') else ''
            multi_name_hand_edited = plugin.find('multi_name_hand_edited').text if plugin.find('multi_name_hand_edited') else '0'

            # Look for an existing operation with the exact same name (and multi-instance name)
            orig_node = None
            for li in seq.find_all('rdf:li'):
                if li.get('darktable:operation') == operation and li.get('darktable:multi_name', '') == multi_name:
                    orig_node = li
                    break

            if orig_node:
                # Update the existing operation in place
                orig_node['darktable:params'] = params
                orig_node['darktable:modversion'] = modversion
                orig_node['darktable:enabled'] = enabled
                orig_node['darktable:blendop_params'] = blendop_params
                orig_node['darktable:blendop_version'] = blendop_version
                orig_node['darktable:multi_priority'] = multi_priority
                orig_node['darktable:multi_name'] = multi_name
                orig_node['darktable:multi_name_hand_edited'] = multi_name_hand_edited

                # Check history_end to ensure this node is active
                if desc and desc.has_attr('darktable:history_end'):
                    orig_num = int(orig_node.get('darktable:num', 0))
                    current_end = int(desc['darktable:history_end'])
                    if current_end <= orig_num:
                        desc['darktable:history_end'] = str(orig_num + 1)
            else:
                # Fabricate a brand new rdf:li node
                highest_num += 1
                new_node = orig_soup.new_tag('rdf:li')
                new_node['darktable:num'] = str(highest_num)
                new_node['darktable:operation'] = operation
                new_node['darktable:modversion'] = modversion
                new_node['darktable:params'] = params
                new_node['darktable:enabled'] = enabled
                new_node['darktable:blendop_params'] = blendop_params
                new_node['darktable:blendop_version'] = blendop_version
                new_node['darktable:multi_priority'] = multi_priority
                new_node['darktable:multi_name'] = multi_name
                new_node['darktable:multi_name_hand_edited'] = multi_name_hand_edited

                seq.append(new_node)

                # Bump the history_end pointer to activate the appended module
                if desc and desc.has_attr('darktable:history_end'):
                    desc['darktable:history_end'] = str(highest_num + 1)

        grainy_xmp_path = original_xmp.with_suffix('.grainy.xmp')
        with open(grainy_xmp_path, 'w') as f:
            f.write(orig_soup.prettify())

        return grainy_xmp_path


    def calculate_noise_level(self, grainy_tif: Path, clean_tif: Path) -> float:
        gmic_bin = self.config['command']['gmic'].strip()
        cmd = f'{gmic_bin} -v 0 "{grainy_tif}" "{clean_tif}" -sub -echo {{iv}} -quit'

        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            output_text = (result.stderr + "\n" + result.stdout).strip()
            lines = [line.strip() for line in output_text.split('\n') if line.strip()]

            if not lines:
                raise ValueError("G'MIC returned no text.")

            val_str = lines[-1].split()[-1]
            variance = float(val_str)
            std_dev = variance ** 0.5

            self.logger.debug(f"Measured Noise Standard Deviation: {std_dev:.2f}")
            return std_dev

        except Exception as e:
            err_msg = (result.stderr + result.stdout).strip() if 'result' in locals() else 'No output'
            self.logger.error(f"Failed to calculate noise: {e} | GMIC raw output: '{err_msg}'")
            return 0.0


    def run_cmd(self, cmd: str):
        self.logger.debug(f"Executing: {cmd}")
        subprocess.run(cmd, shell=True, check=True)


    def process_image(self, img_path: str):
        path = Path(img_path)
        if not path.exists() or path.suffix.upper() not in VALID_EXTENSIONS:
            self.logger.warning(f"Skipping invalid/non-RAW: {img_path}")
            return

        xmp_path = Path(f"{img_path}.xmp")

        rating = self._get_rating(xmp_path)
        allowed_ratings = str(self.args['--rating'])
        if rating not in allowed_ratings:
            self.logger.info(f"Skipping {path.name}: Rating ({rating}) not in allowed filter [{allowed_ratings}]")
            return

        self.logger.info(f"\nProcessing: {path.name} (Rating: {rating})")

        # Proactively outline all possible temporary files
        s1_tif = path.with_name(f"{path.stem}_s1.tif")
        denoised_tiff = path.with_name(f"{path.stem}_s1_denoised.tiff")
        s2_tif = path.with_name(f"{path.stem}_s2.tif")
        s2_deblurred_tif = path.with_name(f"{path.stem}_s2_deblurred.tif")
        full_grainy_tif = path.with_name(f"{path.stem}_full_grainy.tif")
        s1_xmp = xmp_path.with_suffix('.s1.xmp')
        s2_xmp = xmp_path.with_suffix('.s2.xmp')
        grainy_xmp = xmp_path.with_suffix('.grainy.xmp')

        all_temp_files = [s1_tif, denoised_tiff, s2_tif, s2_deblurred_tif, full_grainy_tif, s1_xmp, s2_xmp, grainy_xmp]

        target_ext = f".{self.args['--ext'].lstrip('.')}"
        final_out = self._get_unique_path(f"{path.stem}{target_ext}")
        tmp_final = final_out.parent / f"tmp_final_{final_out.name.replace(' ', '_')}"

        try:
            # Orphan cleanup
            for f in all_temp_files:
                f.unlink(missing_ok=True)

            dt_bin = self.config['command']['darktable'].strip()

            # 1. Stage 1 Export
            s1_xmp_path = self.modify_xmp(xmp_path, stage=1)
            self.run_cmd(f'{dt_bin} "{path}" "{s1_xmp_path}" "{s1_tif}" --apply-custom-presets 0 --core --conf plugins/imageio/format/tiff/bpp=32')

            # 2. NIND Denoise
            nind_bin = self.config['command']['nind_denoise']
            nind_params = self.config['command']['nind_denoise_params']
            self.run_cmd(f'{nind_bin} {nind_params} --input "{s1_tif}" --output "{denoised_tiff}"')
            self._copy_meta(path, denoised_tiff)

            # 3. Stage 2 Export
            s2_xmp_path = self.modify_xmp(xmp_path, stage=2)
            self.run_cmd(f'{dt_bin} "{denoised_tiff}" "{s2_xmp_path}" "{s2_tif}" --icc-intent PERCEPTUAL --icc-type SRGB --apply-custom-presets 0 --core --conf plugins/imageio/format/tiff/bpp=16')

            current_clean = s2_tif
            gmic_bin = self.config['command'].get('gmic', '').strip()

            # 4. RL-Deblur (Optional)
            if not self.args['--no-rldeblur'] and gmic_bin:
                self.logger.info("Applying RL-Deblur...")
                gmic_cmd = (f'{gmic_bin} "{s2_tif}" -deblur_richardsonlucy {self.args["--sigma"]},{self.args["--iter"]},1 '
                            f'-c 0,65535 -o "{s2_deblurred_tif}"')
                self.run_cmd(gmic_cmd)
                current_clean = s2_deblurred_tif

            # 5. Natural Grain Generation & Blending (Optional via Config)
            grain_style = self.config.get('grain', 'style', fallback=None)

            if grain_style and Path(grain_style).exists():
                self.logger.info("Generating natural grain reference via .dtstyle injection...")
                template_style_path = Path(grain_style)
                grainy_xmp_path = self.create_grainy_xmp(xmp_path, template_style_path)

                self.run_cmd(f'{dt_bin} "{path}" "{grainy_xmp_path}" "{full_grainy_tif}" --apply-custom-presets 0 --core --conf plugins/imageio/format/tiff/bpp=16')

                noise_thresh = self.config.getfloat('grain', 'noise_thresh', fallback=150.0)
                noise_amount = self.calculate_noise_level(full_grainy_tif, s2_tif)

                # --- DYNAMIC ALPHA EVALUATION ---
                alpha_raw = self.config.get('grain', 'alpha', fallback='0.20')
                try:
                    # Provide safe math functions and our dynamic 'noise' variable
                    allowed_math = {"min": min, "max": max, "noise": noise_amount}
                    # Evaluate the string from the INI file safely
                    alpha = float(eval(alpha_raw, {"__builtins__": {}}, allowed_math))

                    # Ensure alpha doesn't accidentally drop below 0 or above 1
                    # due to a crazy formula
                    alpha = max(0.0, min(1.0, alpha))

                except Exception as e:
                    self.logger.error(f"Invalid alpha formula '{alpha_raw}': {e}. Falling back to 0.20")
                    alpha = 0.20
                # --------------------------------

                if noise_amount < noise_thresh:
                    self.logger.info(f"Noise level ({noise_amount:.2f}) is below threshold ({noise_thresh}). Skipping grain blend.")
                    gmic_cmd = f'{gmic_bin} "{current_clean}" -/ 256 cut 0,255 round -o "{tmp_final},{self.args["--quality"]}"'
                    self.run_cmd(gmic_cmd)
                else:
                    self.logger.info(f"High noise detected ({noise_amount:.2f}). Blending organic grain (alpha={alpha:.3f})...")
                    blend_cmd = (f'{gmic_bin} "{current_clean}" "{full_grainy_tif}" "{s2_tif}" '
                                 f'-sub[1,2] -mul[1] {alpha} -add[0,1] -c 0,65535 '
                                 f'-/ 256 cut 0,255 round -o "{tmp_final},{self.args["--quality"]}"')
                    self.run_cmd(blend_cmd)
            else:
                if grain_style:
                    self.logger.warning(f"Grain style dtstyle not found: {grain_style}. Skipping grain blend.")

                self.logger.info("Exporting final image...")
                gmic_cmd = f'{gmic_bin} "{current_clean}" -/ 256 cut 0,255 round -o "{tmp_final},{self.args["--quality"]}"'
                self.run_cmd(gmic_cmd)

            # Move to final destination and embed EXIF data from s1_tif
            shutil.move(str(tmp_final), str(final_out))
            self._copy_meta(s1_tif, final_out)
            self.logger.info(f"Successfully exported to: {final_out}")

        finally:
            if not self.args['--debug']:
                self._cleanup(all_temp_files)

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

    args['--quality'] = int(args['--quality'])
    args['--sigma'] = float(args['--sigma'])
    args['--iter'] = int(args['--iter'])

    pipeline = DenoisePipeline(args)
    for f in args['<filenames>']:
        pipeline.process_image(f)

if __name__ == "__main__":
    main()