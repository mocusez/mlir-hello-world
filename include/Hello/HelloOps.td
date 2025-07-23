#ifndef HELLO_OPS
#define HELLO_OPS

include "HelloDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def WorldOp : Hello_Op<"world", [Pure]> {
    let summary = "print Hello, World";
    let description = [{
        The "world" operation prints "Hello, World", and produces
        no results.
    }];
}

def Dict_CreateOp : Hello_Op<"dict.create", [Pure]> {
  let summary = "Create a new dict<string,i32>";
  let results = (outs DictType:$dict);
  let assemblyFormat = "attr-dict `:` type($dict)";
}

def Dict_FreeOp : Hello_Op<"dict.free", []> {
  let summary = "Free the dict<string,i32> memory";
  let arguments = (ins DictType:$dict);
  let assemblyFormat = "$dict attr-dict `:` type($dict)";
}

def Dict_PutOp : Hello_Op<"dict.put", []> {
  let summary = "Insert string->i32";
  let arguments = (ins DictType:$dict, StrAttr:$key, I32Attr:$value);
  let results = (outs DictType:$out);
  let assemblyFormat = "$dict `,` $key `=` $value attr-dict `:` type($dict) `->` type($out)";
}

def Dict_GetOp : Hello_Op<"dict.get", []> {
  let summary = "Lookup string->i32, returns i32";
  let arguments = (ins DictType:$dict, StrAttr:$key);
  let results = (outs I32:$value);
  let assemblyFormat = "$dict `,` $key attr-dict `:` type($dict) `->` type($value)";
}

def Dict_DeleteOp : Hello_Op<"dict.delete", []> {
  let summary = "Delete key string";
  let arguments = (ins DictType:$dict, StrAttr:$key);
  let results = (outs DictType:$out);
  let assemblyFormat = "$dict `,` $key attr-dict `:` type($dict) `->` type($out)";
}

#endif // HELLO_OPS
