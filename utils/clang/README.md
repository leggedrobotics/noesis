## Clang Format

The repository is based on the Google coding guidelines. To ensure, the code follows
the same style and formatting, we use the tool [__Clang-Format__](https://clang.llvm.org/docs/ClangFormat.html).
The configuration settings for Clang is present in the top directory. 

To run the Clang-Format follow the instructions below:
```bash
# install clang-format
sudo apt-get install clang-format
# make the script executable
chmod +x format.sh
# run on all the C++ files in the directory
./format.sh
```
