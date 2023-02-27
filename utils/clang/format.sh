#!/usr/bin/env bash

# This script reformats source files using the clang-format utility.
#
# The file .clang-format in this directory specifies the formatting parameters.
#
# Files are changed in-place, so make sure you don't have anything open in an
# editor, and you may want to commit before formatting in case of awryness.

# Check for path to NOESIS_ROOT
if [ -d "$NOESIS_ROOT" ]; then
    ## declare an array variable
    declare -a directories=("noesis" "noesis_environments" "noesis_examples")

    for DIRECTORY in "${directories[@]}"
    do
        echo "Formatting code under $NOESIS_ROOT/$DIRECTORY"
        find "$NOESIS_ROOT/$DIRECTORY" \( -name '*.hpp' -or -name '*.cpp' -or -name '*.cc' -or -name '*.cxx' -or -name '*.tpp' \) -print0 | xargs -0 clang-format -style=file -i
    done
else
  echo "Path of the environment variable NOESIS_ROOT has not been set!"
fi
