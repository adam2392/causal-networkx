#!/bin/bash -ef

# this is only relevant for GitHub Actions, but it avoids
# https://github.com/actions/virtual-environments/issues/323
# via
# https://github.community/t/ubuntu-latest-apt-repository-list-issues/17182/10#M4501
for apt_file in `grep -lr microsoft /etc/apt/sources.list.d/`; do
    echo "Removing $apt_file"
    sudo rm $apt_file
done

# This also includes the libraries necessary for PyQt5/PyQt6
sudo apt update