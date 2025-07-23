#pragma once

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
namespace hello {
struct DictTypeStorage;
}

#include "Hello/HelloOpsDialect.h.inc"

class DictType : public mlir::Type::TypeBase<DictType, mlir::Type,
                                             hello::DictTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `DictType` with the given key and value types.
  static DictType get(mlir::Type keyType, mlir::Type valueType);

  /// Returns the key type of this dict type.
  mlir::Type getKeyType();

  /// Returns the value type of this dict type.
  mlir::Type getValueType();

  /// The name of this dict type.
  static constexpr mlir::StringLiteral name = "hello.dict";
};
