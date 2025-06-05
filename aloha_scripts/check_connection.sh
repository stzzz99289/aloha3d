#!/bin/bash

# check camera connection
echo "connected cameras: "
rs-enumerate-devices | grep -E '(Name|Serial Number)'

# check arm connection
echo "connected arms: "
ls /dev/ | grep ttyDXL