#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

#include "Hello/HelloDialect.h"
#include "Hello/HelloPasses.h"
#include "Hello/HelloOps.h"

namespace cl = llvm::cl;
namespace {
enum Action {
  None,
  DumpMLIR,
  DumpMLIRLLVM,
  DumpLLVMIR,
};
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump"))
);

int runJit(mlir::ModuleOp module,bool dumpIr=false) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create target machine and configure the LLVM Module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(3 , /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  if(dumpIr){
    llvm::outs() << *llvmModule << "\n";
  } else {
    llvm::ExitOnError ExitOnErr;
    auto J = ExitOnErr(llvm::orc::LLJITBuilder().create());
    ExitOnErr(J->addIRModule(llvm::orc::ThreadSafeModule(std::move(llvmModule), std::make_unique<llvm::LLVMContext>())));
    auto MainSymbol = ExitOnErr(J->lookup("main"));
    auto *main1 = MainSymbol.toPtr<int()>();
    main1();
  }

  return 0;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "Hello compiler\n");
  // Init
  mlir::MLIRContext context;
  context.getOrLoadDialect<hello::HelloDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();

  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
  
  // Generate Code Block
  auto funcType = builder.getFunctionType({}, {});
  auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", funcType);

  std::vector<mlir::Type> elementTypes;
  auto dataType1 = mlir::RankedTensorType::get({3}, builder.getI64Type());
  auto dataType2 = mlir::RankedTensorType::get({3}, builder.getF64Type());
  elementTypes.push_back(dataType1);
  elementTypes.push_back(dataType2);
  StructType structType = StructType::get(elementTypes);
  llvm::outs() << "MyStructType: " << structType << "\n";

  module->push_back(func);
  auto entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // std::vector<int32_t> values = {1, 2, 3};
  // auto valueAttr = builder.getI32ArrayAttr(values);
  std::vector<int64_t> intValues = {1, 2, 3};
  std::vector<double> floatValues = {1.1, 2.1, 3.1};
  auto intAttr = mlir::DenseElementsAttr::get(
      dataType1, 
      llvm::ArrayRef<int64_t>(intValues)
  );
  auto floatAttr = mlir::DenseElementsAttr::get(
      dataType2, 
      llvm::ArrayRef<double>(floatValues)
  );
  std::vector<mlir::Attribute> values = {intAttr, floatAttr};
  auto valueAttr = builder.getArrayAttr(values);
  builder.create<hello::StructConstantOp>(builder.getUnknownLoc(), structType,valueAttr);
  builder.create<hello::WorldOp>(builder.getUnknownLoc());

  builder.create<mlir::func::ReturnOp>(
      builder.getUnknownLoc()); 

  module->print(llvm::outs());
  // if (emitAction == Action::DumpMLIR){
  //   module->print(llvm::outs());
  //   return 0;
  // }

  // // Add Pass for lowering to LLVM
  // mlir::PassManager passManager(&context);
  // passManager.addPass(hello::createLowerToLLVMPass());

  // if (mlir::failed(passManager.run(*module))) {
  //   return 4;
  // }

  // if (emitAction == Action::DumpMLIRLLVM){
  //   module->print(llvm::outs());
  // } else if(emitAction == Action::DumpLLVMIR){
  //   runJit(*module,true);
  // } else {
  //   runJit(*module);
  // }

  return 0;
}
