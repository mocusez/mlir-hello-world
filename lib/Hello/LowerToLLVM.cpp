#include "Hello/HelloDialect.h"
#include "Hello/HelloOps.h"
#include "Hello/HelloPasses.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include <iostream>

namespace hello {
class WorldOpLowering : public mlir::ConversionPattern {
public:
  explicit WorldOpLowering(mlir::MLIRContext *context)
      : mlir::ConversionPattern(hello::WorldOp::getOperationName(), 1,
                                context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    auto loc = op->getLoc();
    mlir::Value helloWorld = getOrCreateGlobalString(
        loc, rewriter, "hello_word_string",
        mlir::StringRef("Hello, World! \n\0", 16), parentModule);

    rewriter.create<mlir::LLVM::CallOp>(loc, getPrintfType(context), printfRef,
                                        helloWorld);
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  static mlir::LLVM::LLVMFunctionType
  getPrintfType(mlir::MLIRContext *context) {
    auto llvmI32Ty = mlir::IntegerType::get(context, 32);
    auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(context);
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                                        /*isVarArg=*/true);
    return llvmFnType;
  }

  static mlir::FlatSymbolRefAttr
  getOrInsertPrintf(mlir::PatternRewriter &rewriter, mlir::ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf")) {
      return mlir::SymbolRefAttr::get(context, "printf");
    }

    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                            getPrintfType(context));
    return mlir::SymbolRefAttr::get(context, "printf");
  }

  static mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                             mlir::OpBuilder &builder,
                                             mlir::StringRef name,
                                             mlir::StringRef value,
                                             mlir::ModuleOp module) {
    // Create the global at the entry of the module.
    mlir::LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    mlir::Value globalPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc, mlir::IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));

    return builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        global.getType(), globalPtr, mlir::ArrayRef<mlir::Value>({cst0, cst0}));
  }
};

class DictCreateOpLowering : public mlir::ConversionPattern {
public:
  explicit DictCreateOpLowering(mlir::TypeConverter &typeConverter,
                                mlir::MLIRContext *context)
      : mlir::ConversionPattern(
            typeConverter, hello::CreateOp::getOperationName(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    auto resultType =
        getTypeConverter()->convertType(op->getResult(0).getType());
    if (!resultType) {
      return mlir::failure();
    }

    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    auto createMapRef = getOrInsertCreateMap(rewriter, module);

    auto callOp = rewriter.create<mlir::LLVM::CallOp>(
        loc, resultType, createMapRef, mlir::ValueRange{});

    rewriter.replaceOp(op, callOp.getResult());
    return mlir::success();
  }

private:
  mlir::FlatSymbolRefAttr getOrInsertCreateMap(mlir::PatternRewriter &rewriter,
                                               mlir::ModuleOp module) const {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("create_map"))
      return mlir::SymbolRefAttr::get(context, "create_map");

    // Create function type: () -> !llvm.ptr
    auto resultType = mlir::LLVM::LLVMPointerType::get(context);
    auto fnType =
        mlir::LLVM::LLVMFunctionType::get(resultType, std::nullopt, false);

    // Insert function declaration
    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "create_map",
                                            fnType);

    return mlir::SymbolRefAttr::get(context, "create_map");
  }
};

class DictFreeOpLowering : public mlir::ConversionPattern {
public:
  explicit DictFreeOpLowering(mlir::MLIRContext *context)
      : mlir::ConversionPattern(hello::FreeOp::getOperationName(), 1, context) {
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto dict = operands[0]; // The dictionary pointer

    // Get module to insert external function declarations
    mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

    // Get or insert free_map function declaration
    auto freeMapRef = getOrInsertFreeMap(rewriter, parentModule);

    // Call free_map(dict)
    rewriter.create<mlir::LLVM::CallOp>(loc,
                                        mlir::TypeRange{}, // void return type
                                        freeMapRef, mlir::ValueRange{dict});

    // Erase the original op (no result)
    rewriter.eraseOp(op);

    return mlir::success();
  }

private:
  mlir::FlatSymbolRefAttr getOrInsertFreeMap(mlir::PatternRewriter &rewriter,
                                             mlir::ModuleOp module) const {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("free_map"))
      return mlir::SymbolRefAttr::get(context, "free_map");

    // Create function type: (ptr) -> void
    auto ptrType = mlir::LLVM::LLVMPointerType::get(context);
    auto voidType = mlir::LLVM::LLVMVoidType::get(context);
    auto fnType = mlir::LLVM::LLVMFunctionType::get(voidType, {ptrType}, false);

    // Insert function declaration
    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "free_map",
                                            fnType);

    return mlir::SymbolRefAttr::get(context, "free_map");
  }
};

class DictGetOpLowering : public mlir::ConversionPattern {
public:
  explicit DictGetOpLowering(mlir::MLIRContext *context)
      : mlir::ConversionPattern(hello::GetOp::getOperationName(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto getOp = llvm::cast<hello::GetOp>(op);

    // Get dict operand
    auto dict = operands[0];

    // Get the key string attribute
    auto keyAttr = getOp.getKey();

    // Get or create global string for the key
    auto keyGlobal =
        getOrCreateGlobalString(loc, rewriter, "key_" + keyAttr.str(), keyAttr,
                                op->getParentOfType<mlir::ModuleOp>());

    // Get module to insert external function declarations
    mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

    // Get or insert get function declaration
    auto getRef = getOrInsertGet(rewriter, parentModule);

    // Call get function: get(map, key)
    auto i32PtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto resultPtr =
        rewriter
            .create<mlir::LLVM::CallOp>(loc, i32PtrType, getRef,
                                        mlir::ValueRange{dict, keyGlobal})
            .getResult();

    // Load the value from the pointer
    auto i32Type = mlir::IntegerType::get(rewriter.getContext(), 32);
    auto result = rewriter.create<mlir::LLVM::LoadOp>(loc, i32Type, resultPtr);

    // Replace the original op with the loaded value
    rewriter.replaceOp(op, result);

    return mlir::success();
  }

private:
  mlir::FlatSymbolRefAttr getOrInsertGet(mlir::PatternRewriter &rewriter,
                                         mlir::ModuleOp module) const {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("get"))
      return mlir::SymbolRefAttr::get(context, "get");

    // Create function type: (ptr, ptr) -> ptr
    auto ptrType = mlir::LLVM::LLVMPointerType::get(context);
    auto fnType =
        mlir::LLVM::LLVMFunctionType::get(ptrType, {ptrType, ptrType}, false);

    // Insert function declaration
    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "get", fnType);

    return mlir::SymbolRefAttr::get(context, "get");
  }

  // Helper for creating string constants - reusing the existing function from
  // WorldOpLowering
  mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                      mlir::OpBuilder &builder,
                                      mlir::StringRef name,
                                      mlir::StringRef value,
                                      mlir::ModuleOp module) const {
    // Create the global at the entry of the module.
    mlir::LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8),
          value.size() + 1); // +1 for null terminator
      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(value.str() + '\0'));
    }

    // Get the pointer to the first character in the global string.
    mlir::Value globalPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc, mlir::IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));

    return builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        global.getType(), globalPtr, mlir::ArrayRef<mlir::Value>({cst0, cst0}));
  }
};

class DictPutOpLowering : public mlir::ConversionPattern {
public:
  explicit DictPutOpLowering(mlir::MLIRContext *context)
      : mlir::ConversionPattern(hello::PutOp::getOperationName(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto putOp = llvm::cast<hello::PutOp>(op);

    // Get input operands
    auto dict = operands[0]; // The dictionary pointer

    // Get the key and value attributes
    auto keyAttr = putOp.getKey();
    auto valueAttr = putOp.getValue();

    // Get or create global string for the key
    auto keyGlobal =
        getOrCreateGlobalString(loc, rewriter, "key_" + keyAttr.str(), keyAttr,
                                op->getParentOfType<mlir::ModuleOp>());

    // Create the integer constant for the value
    auto valueConst = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, mlir::IntegerType::get(rewriter.getContext(), 32),
        rewriter.getI32IntegerAttr(valueAttr));

    // Get module to insert external function declarations
    mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

    // Get or insert put function declaration
    auto putRef = getOrInsertPut(rewriter, parentModule);

    // Call put function: put(map, key, value)
    rewriter.create<mlir::LLVM::CallOp>(
        loc, mlir::TypeRange{}, // void return type
        putRef, mlir::ValueRange{dict, keyGlobal, valueConst});

    // Replace the original op with the input dictionary since put returns the
    // same dict
    rewriter.replaceOp(op, dict);

    return mlir::success();
  }

private:
  mlir::FlatSymbolRefAttr getOrInsertPut(mlir::PatternRewriter &rewriter,
                                         mlir::ModuleOp module) const {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("put"))
      return mlir::SymbolRefAttr::get(context, "put");

    // Create function type: (ptr, ptr, i32) -> void
    auto ptrType = mlir::LLVM::LLVMPointerType::get(context);
    auto i32Type = mlir::IntegerType::get(context, 32);
    auto voidType = mlir::LLVM::LLVMVoidType::get(context);

    auto fnType = mlir::LLVM::LLVMFunctionType::get(
        voidType, {ptrType, ptrType, i32Type}, false);

    // Insert function declaration
    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "put", fnType);

    return mlir::SymbolRefAttr::get(context, "put");
  }

  // Helper for creating string constants - sharing the implementation
  mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                      mlir::OpBuilder &builder,
                                      mlir::StringRef name,
                                      mlir::StringRef value,
                                      mlir::ModuleOp module) const {
    // Create the global at the entry of the module.
    mlir::LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8),
          value.size() + 1); // +1 for null terminator
      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(value.str() + '\0'));
    }

    // Get the pointer to the first character in the global string.
    mlir::Value globalPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc, mlir::IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));

    return builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        global.getType(), globalPtr, mlir::ArrayRef<mlir::Value>({cst0, cst0}));
  }
};

class DictDeleteOpLowering : public mlir::ConversionPattern {
public:
  explicit DictDeleteOpLowering(mlir::MLIRContext *context)
      : mlir::ConversionPattern(hello::DeleteOp::getOperationName(), 1,
                                context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto deleteOp = llvm::cast<hello::DeleteOp>(op);

    // Get input operands
    auto dict = operands[0]; // The dictionary pointer

    // Get the key attribute
    auto keyAttr = deleteOp.getKey();

    // Get or create global string for the key
    auto keyGlobal =
        getOrCreateGlobalString(loc, rewriter, "key_" + keyAttr.str(), keyAttr,
                                op->getParentOfType<mlir::ModuleOp>());

    // Get module to insert external function declarations
    mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

    // Get or insert delete function declaration
    auto deleteRef = getOrInsertDelete(rewriter, parentModule);

    // Call delete function: delete(map, key)
    rewriter.create<mlir::LLVM::CallOp>(
        loc, mlir::TypeRange{}, // void return type
        deleteRef, mlir::ValueRange{dict, keyGlobal});

    // Replace the original op with the input dictionary
    rewriter.replaceOp(op, dict);

    return mlir::success();
  }

private:
  mlir::FlatSymbolRefAttr getOrInsertDelete(mlir::PatternRewriter &rewriter,
                                            mlir::ModuleOp module) const {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("delete"))
      return mlir::SymbolRefAttr::get(context, "delete");

    // Create function type: (ptr, ptr) -> void
    auto ptrType = mlir::LLVM::LLVMPointerType::get(context);
    auto voidType = mlir::LLVM::LLVMVoidType::get(context);

    auto fnType =
        mlir::LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType}, false);

    // Insert function declaration
    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "delete", fnType);

    return mlir::SymbolRefAttr::get(context, "delete");
  }

  // Helper for creating string constants
  mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                      mlir::OpBuilder &builder,
                                      mlir::StringRef name,
                                      mlir::StringRef value,
                                      mlir::ModuleOp module) const {
    // Same implementation as in other classes
    // ...as in the previous implementations...
    mlir::LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);
      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(value.str() + '\0'));
    }

    mlir::Value globalPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc, mlir::IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));

    return builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        global.getType(), globalPtr, mlir::ArrayRef<mlir::Value>({cst0, cst0}));
  }
};

} // namespace hello

namespace {
class HelloToLLVMLoweringPass
    : public mlir::PassWrapper<HelloToLLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HelloToLLVMLoweringPass)
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect,
                    mlir::cf::ControlFlowDialect>();
  }

  void runOnOperation() final;
};
} // namespace

void HelloToLLVMLoweringPass::runOnOperation() {
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  mlir::LLVMTypeConverter typeConverter(&getContext());
  typeConverter.addConversion([](DictType type) -> mlir::Type {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });

  mlir::RewritePatternSet patterns(&getContext());

  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  patterns.add<hello::WorldOpLowering>(&getContext());
  patterns.add<hello::DictCreateOpLowering>(typeConverter, &getContext());
  patterns.add<hello::DictFreeOpLowering>(&getContext());
  patterns.add<hello::DictPutOpLowering>(&getContext());
  patterns.add<hello::DictGetOpLowering>(&getContext());
  patterns.add<hello::DictDeleteOpLowering>(&getContext());

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> hello::createLowerToLLVMPass() {
  return std::make_unique<HelloToLLVMLoweringPass>();
}
