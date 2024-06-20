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

# Install necessary libraries and dependencies
sudo apt-get install -y python3-pip python3-dev build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev

# Install JAX, Haiku, TensorFlow, and other required libraries
pip3 install jax jaxlib haiku tensorflow tensorflow-text

# Clean up
sudo apt-get autoremove -y
sudo apt-get clean
