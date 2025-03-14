#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <video_directory> <output_directory>"
    exit 1
fi

# Assign input arguments to variables
VIDEO_DIR="$1"
OUTPUT_DIR="$2"
OUTPUT_FILE="$OUTPUT_DIR/merged_video.mp4"

# Ensure the video directory exists
if [ ! -d "$VIDEO_DIR" ]; then
    echo "Error: Video directory does not exist!"
    exit 1
fi

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Generate the file list for ffmpeg
FILE_LIST="$OUTPUT_DIR/file_list.txt"
rm -f "$FILE_LIST"
touch "$FILE_LIST"

# Find and sort all .mp4 files in the directory, then add them to the list
for file in $(ls "$VIDEO_DIR"/*.mp4 | sort); do
    echo "file '$file'" >> "$FILE_LIST"
done

# Check if the file list is empty
if [ ! -s "$FILE_LIST" ]; then
    echo "Error: No MP4 files found in $VIDEO_DIR!"
    exit 1
fi

# Concatenate videos using ffmpeg
ffmpeg -f concat -safe 0 -i "$FILE_LIST" -c copy "$OUTPUT_FILE"

# Cleanup the temporary file list
rm -f "$FILE_LIST"

echo "Merged video saved as: $OUTPUT_FILE"
