#!/bin/bash
#echo "Downloading Knowledge Bases..."
#curl "https://files.de-1.osf.io/v1/resources/8mqs4/providers/osfstorage/?view_only=282c38027c8043d5abd76a98001c31fa&zip=" --output supporting_files/knowledge.zip
#echo "Knowledge downloaded, please extract the 'knowledge.zip' within supporting_files, and extract edges.csv.gz within 'ConceptNet' folder"

URL="https://files.de-1.osf.io/v1/resources/8mqs4/providers/osfstorage/69398348ba8775a8380a33f2/?zip="
DOWNLOAD_DIR="supporting_files"
ZIP_FILE="$DOWNLOAD_DIR/knowledge.zip"
EXTRACT_DIR="$DOWNLOAD_DIR/kb"

echo "Creating directory: $DOWNLOAD_DIR"
mkdir -p "$DOWNLOAD_DIR"

echo "Downloading zip file from $URL..."
curl -L "$URL" --output "$ZIP_FILE"
echo "Download complete"

echo "Creating extraction directory: $EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"

echo "Extracting $ZIP_FILE to $EXTRACT_DIR..."
unzip -oq "$ZIP_FILE" -d "$EXTRACT_DIR"
echo "Extraction complete"

echo "Processing and renaming files..."
for path in "$EXTRACT_DIR"/*; do
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

            final_path="$EXTRACT_DIR/$new_filename"
            mv "$file_to_process" "$final_path"
            echo "Processed and moved '$new_filename' to $EXTRACT_DIR"
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

echo "Cleaning up empty folders"
find "$EXTRACT_DIR" -mindepth 1 -type d -empty -delete
echo "Cleanup complete"

echo "Knowledge downloaded"
