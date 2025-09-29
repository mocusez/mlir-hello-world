# MLIR：Hello World

Print “Hello World!” from a MLIR Program

## How to use

```
git clone https://github.com/mocusez/mlir-hello-world
```

Setup CMake MLIR environment on Debian-sid with MLIR Environment -> [CMake_MLIR_Toy](https://github.com/mocusez/CMake_MLIR_Toy)

```
mkdir build
cmake -B build -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang-20 -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++-20 --no-warn-unused-cli -G Ninja
cmake  --build build --config Debug --target helloworld
cmake  --build build --config Debug --target hello-opt
```

then

```bash
./build/bin/helloworld
```

it will show：

```
Hello, World!
```



## Using .mlir file

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
(base) root@46f815dd64f4:~/mlir-hello-world/test# ../build/bin/hello-opt hello_world.mlir -emit=llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@str = private unnamed_addr constant [15 x i8] c"Hello, World! \00", align 1

; Function Attrs: nofree nounwind
define void @main() local_unnamed_addr #0 {
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #0

attributes #0 = { nofree nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```



