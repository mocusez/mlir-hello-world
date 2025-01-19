#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace hello {
    std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
} // namespace hello
