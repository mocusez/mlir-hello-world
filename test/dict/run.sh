../../build/bin/hello-opt dict.mlir -emit=llvm > dict.ll
${CC:-clang} dict.ll -o dict
./dict
rm -f dict
rm -f dict.ll