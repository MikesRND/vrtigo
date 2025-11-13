#!/bin/bash
# Make all .sh files in the scripts directory executable

find scripts/ -name "*.sh" -type f -exec chmod +x {} \;

echo "All .sh files in scripts/ are now executable"
