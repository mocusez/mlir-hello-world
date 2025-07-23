// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "Hello/HelloDialect.h"
#include "Hello/HelloOps.h"

using namespace mlir;
using namespace hello;

//===----------------------------------------------------------------------===//
// Hello dialect.
//===----------------------------------------------------------------------===//

#include "Hello/HelloOpsDialect.cpp.inc"

void HelloDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Hello/HelloOps.cpp.inc"
      >();
  addTypes<DictType>();
}

namespace hello {
struct DictTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` defines what uniquely identifies this type.
  /// For dict type, we unique on the key type and value type pair.
  using KeyTy = std::pair<mlir::Type, mlir::Type>;

  /// Constructor for the type storage instance.
  DictTypeStorage(mlir::Type keyType, mlir::Type valueType)
      : keyType(keyType), valueType(valueType) {}

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key.first == keyType && key.second == valueType;
  }

  /// Define a hash function for the key type.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  /// Define a construction function for the key type.
  static KeyTy getKey(mlir::Type keyType, mlir::Type valueType) {
    return KeyTy(keyType, valueType);
  }

  /// Define a construction method for creating a new instance of this storage.
  static DictTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    // Allocate the storage instance and construct it.
    return new (allocator.allocate<DictTypeStorage>())
        DictTypeStorage(key.first, key.second);
  }

  /// The key and value types of the dict.
  mlir::Type keyType;
  mlir::Type valueType;
};
} // namespace hello

DictType DictType::get(mlir::Type keyType, mlir::Type valueType) {
  return Base::get(keyType.getContext(), keyType, valueType);
}

mlir::Type DictType::getKeyType() { return getImpl()->keyType; }

/// Returns the value type of this dict type.
mlir::Type DictType::getValueType() { return getImpl()->valueType; }

mlir::Type HelloDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef typeTag;
  if (parser.parseKeyword(&typeTag))
    return mlir::Type();

  if (typeTag == "dict") {
    if (parser.parseLess())
      return mlir::Type();

    mlir::Type keyType;
    if (parser.parseType(keyType))
      return mlir::Type();

    if (parser.parseComma())
      return mlir::Type();

    mlir::Type valueType;
    if (parser.parseType(valueType))
      return mlir::Type();

    if (parser.parseGreater())
      return mlir::Type();

    return DictType::get(keyType, valueType);
  }

  parser.emitError(parser.getNameLoc(), "unknown hello type: ") << typeTag;
  return mlir::Type();
}

void HelloDialect::printType(mlir::Type type,
                             mlir::DialectAsmPrinter &printer) const {
  if (auto dictType = mlir::dyn_cast<DictType>(type)) {
    printer << "dict<";
    printer.printType(dictType.getKeyType());
    printer << ", ";
    printer.printType(dictType.getValueType());
    printer << ">";
    return;
  }

  llvm_unreachable("unhandled hello type");
}
