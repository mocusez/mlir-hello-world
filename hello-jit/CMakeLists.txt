set(LIBS
        MLIRHello
        MLIR
        LLVM
        MLIRExecutionEngineShared
        )
add_llvm_executable(helloworld hello_world.cpp)

llvm_update_compile_flags(helloworld)
target_link_libraries(helloworld PRIVATE ${LIBS})
