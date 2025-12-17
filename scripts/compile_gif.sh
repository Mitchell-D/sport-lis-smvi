#!/bin/bash

## example usage:
## ./compile_gif.sh "figures/daily/smvi_binary_EastTN_*_soilm-200.png" figures/gifs/smvi_binary_EastTN_soilm-200.gif

## jon's gif format
## convert -delay 75 figures/daily/smvi_pixelwise-percentile_*soilm-100.png \( figures/daily/smvi_pixelwise-percentile_la_20231031_soilm-100.png -delay 250 \) -loop 0 sportlis_smvi_la_20230606-20231031_anim_mitchell-pixelwise.gif

## defaults
FPS=8
COLORS=256

## parse options
while getopts "f:c:" opt; do
  case $opt in
    f) FPS="$OPTARG" ;;
    c) COLORS="$OPTARG" ;;
    *) echo "Usage: $0 [-f fps] [-c colors] <input_glob> <output_gif>"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

## required args
INPUT_GLOB="$1"
OUTPUT_GIF="$2"

if [ -z "$INPUT_GLOB" ] || [ -z "$OUTPUT_GIF" ]; then
  echo "Usage: $0 [-f fps] [-c colors] <input_glob> <output_gif>"
  exit 1
fi

## convert fps to ImageMagick delay (1/100ths of a second)
DELAY=$((100 / FPS))

# Build GIF
convert -delay "$DELAY" -loop 0 -colors "$COLORS" $INPUT_GLOB "$OUTPUT_GIF"
