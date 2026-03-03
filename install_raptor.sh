#!/bin/bash

# Define the installation directory inside the current project folder
INSTALL_DIR="$(pwd)/raptor"

# Check if it's already installed so we don't waste time recompiling
if [ -f "$INSTALL_DIR/bin/rapper" ]; then
    echo "Rapper is already installed at $INSTALL_DIR/bin/rapper"
    exit 0
fi

echo "Downloading Raptor2 source code..."
wget -q http://download.librdf.org/source/raptor2-2.0.16.tar.gz
tar -xzf raptor2-2.0.16.tar.gz
cd raptor2-2.0.16

echo "Configuring and compiling to $INSTALL_DIR..."
./configure --prefix="$INSTALL_DIR"
make
make install

echo "Cleaning up temporary build files..."
cd ..
rm -rf raptor2-2.0.15 raptor2-2.0.15.tar.gz

echo "Success! Rapper is ready to use at: $INSTALL_DIR/bin/rapper"