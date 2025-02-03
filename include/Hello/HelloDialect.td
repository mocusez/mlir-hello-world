#ifndef HELLO_DIALECT
#define HELLO_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"


//===----------------------------------------------------------------------===//
// Hello dialect definition.
//===----------------------------------------------------------------------===//

def Hello_Dialect : Dialect {
    let name = "hello";
    let summary = "A hello out-of-tree MLIR dialect.";
    let description = [{
        This dialect is minimal example to implement hello-world kind of sample code
        for MLIR.
    }];
    let cppNamespace = "::hello";
    // let hasConstantMaterializer = 1;
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base hello operation definition.
//===----------------------------------------------------------------------===//

class Hello_Op<string mnemonic, list<Trait> traits = []> :
        Op<Hello_Dialect, mnemonic, traits>;

def Toy_StructType :
    DialectType<Hello_Dialect, CPred<"::llvm::isa<StructType>($_self)">,
                "Toy struct type">;

def Toy_Type : AnyTypeOf<[F64Tensor, Toy_StructType]>;

// class Hello_Type<string name, string typeMnemonic, list<Trait> traits = []>
//     : TypeDef<Hello_Dialect, name, traits> {
//   let mnemonic = typeMnemonic;
// }

// def MyStructType : Hello_Type<"MyStruct", "mystruct"> {
//   let summary = "A type representing a struct with 5 int32 values";

//   // 定义5个int32值的存储
//   let parameters = (ins "int":$value1, "int":$value2, "int":$value3, "int":$value4, "int":$value5);

//   // 定义文本格式
//   let assemblyFormat = "`<` $value1 `,` $value2 `,` $value3 `,` $value4 `,` $value5 `>`";
// }

#endif // HELLO_DIALECT
