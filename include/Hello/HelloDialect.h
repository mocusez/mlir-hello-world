#pragma once

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

#include "Hello/HelloOpsDialect.h.inc"

struct StructTypeStorage;

class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               StructTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be at least one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes();

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }

  static constexpr mlir::StringLiteral name = "hello.struct";
};
