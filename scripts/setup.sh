#!/bin/bash

read -rp "Enter scene id: " id

source_root="/coc/flash5/datasets/scannetpp/data/data"
source_scene="${source_root}/${id}"
destination="./dataset/${id}"

images="${source_scene}/dslr/resized_undistorted_images"
transforms="${source_scene}/dslr/nerfstudio/transforms_undistorted.json"

if [[ ! -d "$images" ]]; then
    echo "Images directory not found: $images" >&2
    exit 1
fi

if [[ ! -f "$transforms" ]]; then
    echo "Transform file not found: $transforms" >&2
    exit 1
fi

mkdir -p "$destination"

cp -R "$images" "$destination/"
cp "$transforms" "$destination/"

echo "Copied scene $id to $destination"
