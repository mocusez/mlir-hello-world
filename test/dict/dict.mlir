module {
  func.func @test_dict_operations() -> i32 {
    %dict0 = hello.dict.create : !hello.dict<index, i32>
    
    %dict1 = hello.dict.put %dict0, "first1" = 100 : !hello.dict<index, i32> -> !hello.dict<index, i32>
    %dict2 = hello.dict.put %dict1, "second" = 200 : !hello.dict<index, i32> -> !hello.dict<index, i32>
    %dict3 = hello.dict.put %dict2, "third" = 300 : !hello.dict<index, i32> -> !hello.dict<index, i32>
    
    %val1 = hello.dict.get %dict3, "first1" : !hello.dict<index, i32> -> i32
    %val2 = hello.dict.get %dict3, "second" : !hello.dict<index, i32> -> i32
    %val3 = hello.dict.get %dict3, "third" : !hello.dict<index, i32> -> i32
    
    %dict4 = hello.dict.delete %dict3, "second" : !hello.dict<index, i32> -> !hello.dict<index, i32>
    
    hello.dict.free %dict4 : !hello.dict<index, i32>

    func.return %val2 : i32
  }
}