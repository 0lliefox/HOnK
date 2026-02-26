#!/bin/bash
set -e

URL="https://files.de-1.osf.io/v1/resources/8mqs4/providers/osfstorage/?view_only=6684c951c52e4f068d0577f1724b6a7c&zip="
DOWNLOAD_DIR="supporting_files"
ZIP_FILE="$DOWNLOAD_DIR/knowledge.zip"
EXTRACT_DIR="$DOWNLOAD_DIR/kb"

echo "Creating directory: $DOWNLOAD_DIR"
mkdir -p "$DOWNLOAD_DIR"

echo "Downloading zip file from $URL..."
curl -f -L "$URL" --output "$ZIP_FILE"
echo "Download complete"

echo "Creating extraction directory: $EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"

echo "Extracting $ZIP_FILE to $EXTRACT_DIR..."
unzip -oq "$ZIP_FILE" -d "$EXTRACT_DIR"
echo "Extraction complete"

# Define the exact path to the Data Sources directory
DATA_SOURCES_DIR="$EXTRACT_DIR/Data Sources"

# Change into the nested directory before processing
echo "Changing into $DATA_SOURCES_DIR..."
cd "$DATA_SOURCES_DIR" || { echo "Error: Data Sources directory not found in zip."; exit 1; }

echo "Processing and renaming files..."
# Loop through the folders inside the Data Sources directory
for path in ./*; do
  # Check if item is a directory
  if [ -d "$path" ]; then
    dir_name=$(basename "$path") # Get name of directory
    file_to_process=$(find "$path" -type f -print -quit)

    # Check if a file was actually found
    if [ -n "$file_to_process" ]; then

      # ConceptNet/Wiktionary/GeoNames are compressed to .gz, so extract
      if [[ "$file_to_process" == *.gz ]]; then
        echo "Extracting gzip file to '$dir_name'"
        gunzip "$file_to_process"
        file_to_process=$(find "$path" -type f -print -quit)
      fi

      file_count=$(find "$path" -type f | wc -l)

      if [ "$file_count" -eq 1 ]; then
        if [ -n "$file_to_process" ]; then
            original_filename=$(basename -- "$file_to_process")
            extension="${original_filename##*.}"

            new_filename="$dir_name"
            if [[ "$original_filename" != "$extension" ]]; then
                new_filename="${dir_name}.${extension}"
            fi

            # Move the file out of the subfolder into the Data Sources root
            final_path="./$new_filename"
            mv "$file_to_process" "$final_path"
            echo "Processed and moved '$new_filename'"
        else
            echo "Warning: No file found after extraction in '$dir_name'"
        fi
      else
        echo "More than one file in folder, leaving as is"
      fi

    else
      echo "Warning: No file found in directory '$path'"
    fi
  fi
done

echo "Cleaning up empty folders..."
find . -mindepth 1 -type d -empty -delete
echo "Internal cleanup complete"

# Return to the base directory before moving/deleting files
cd - > /dev/null

echo "Finalizing directory structure..."
# Move all processed files from Data Sources into the main kb directory
mv "$DATA_SOURCES_DIR"/* "$EXTRACT_DIR"/

# Remove the now-empty Data Sources directory and the unwanted Results directory
rm -rf "$DATA_SOURCES_DIR"
rm -rf "$EXTRACT_DIR/Results"

echo "Knowledge downloaded, processed, and cleaned up successfully!"