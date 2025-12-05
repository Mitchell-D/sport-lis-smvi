#!/bin/bash

## example usage:
## ./compile_gif.sh "figures/daily/smvi_binary_EastTN_*_soilm-200.png" figures/gifs/smvi_binary_EastTN_soilm-200.gif

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
