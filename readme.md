# MLIR: Hello World

Print "Hello World!" from an MLIR Program.

## How to use

```bash
git clone https://github.com/mocusez/mlir-hello-world
cd mlir-hello-world
```

### Setup Environment

Install [pixi](https://pixi.sh) to manage the MLIR/LLVM toolchain:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
pixi install
```

### Build

```bash
pixi run cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja
pixi run cmake --build build
```

### Run

```bash
./build/bin/helloworld
```

Output:

```
Hello, World!
```

## Using .mlir Files

### Output LLVM dialect

```bash
./build/bin/hello-opt test/hello_world.mlir -emit=mlir-llvm
```

Result:

```mlir
module {
  llvm.mlir.global internal constant @hello_word_string("Hello, World! \0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @main() {
    %0 = llvm.mlir.addressof @hello_word_string : !llvm.ptr
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.getelementptr %0[%1, %1] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<16 x i8>
    %3 = llvm.call @printf(%2) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    llvm.return
  }
}
```

### Output LLVM IR

```bash
./build/bin/hello-opt test/hello_world.mlir -emit=llvm
```

Result:

```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@hello_word_string = internal constant [16 x i8] c"Hello, World! \0A\00"

declare i32 @printf(ptr, ...)

define void @main() {
  %1 = getelementptr [16 x i8], ptr @hello_word_string, i64 0, i64 0
  %2 = call i32 (ptr, ...) @printf(ptr %1)
  ret void
}
```
