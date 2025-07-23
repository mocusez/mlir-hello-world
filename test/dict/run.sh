../../build/bin/hello-opt dict.mlir -emit=llvm > dict.ll
clang-20 dict.c dict.ll -o dict
./dict
rm -f dict
rm -f dict.ll