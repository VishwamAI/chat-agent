#!/bin/bash

# Update package lists
sudo apt-get update

# Upgrade installed packages without prompting for Docker daemon restart
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# Install Python dependencies from requirements.txt
pip3 install -r requirements.txt

# Check for updates and install them if available without prompting for Docker daemon restart
sudo DEBIAN_FRONTEND=noninteractive apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# Download the latest Google Chrome package
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

# Install the package
sudo dpkg -i google-chrome-stable_current_amd64.deb

# Fix any dependency issues
sudo apt-get install -f

# Clean up
rm google-chrome-stable_current_amd64.deb
