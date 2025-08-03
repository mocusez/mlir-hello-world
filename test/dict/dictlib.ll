; ModuleID = 'dict.c'
source_filename = "dict.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.HashMap = type { [100 x ptr] }
%struct.Entry = type { ptr, i32, ptr }

@.str = private unnamed_addr constant [29 x i8] c"Starting dictionary test...\0A\00", align 1
@.str.1 = private unnamed_addr constant [38 x i8] c"Result from test_dict_operations: %d\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @hash(ptr noundef %0) alwaysinline #0 {
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  %4 = alloca i32, align 4
  store ptr %0, ptr %2, align 8
  store i64 5381, ptr %3, align 8
  br label %5

5:                                                ; preds = %11, %1
  %6 = load ptr, ptr %2, align 8
  %7 = getelementptr inbounds nuw i8, ptr %6, i32 1
  store ptr %7, ptr %2, align 8
  %8 = load i8, ptr %6, align 1
  %9 = sext i8 %8 to i32
  store i32 %9, ptr %4, align 4
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %11, label %19

11:                                               ; preds = %5
  %12 = load i64, ptr %3, align 8
  %13 = shl i64 %12, 5
  %14 = load i64, ptr %3, align 8
  %15 = add i64 %13, %14
  %16 = load i32, ptr %4, align 4
  %17 = sext i32 %16 to i64
  %18 = add i64 %15, %17
  store i64 %18, ptr %3, align 8
  br label %5, !llvm.loop !6

19:                                               ; preds = %5
  %20 = load i64, ptr %3, align 8
  %21 = urem i64 %20, 100
  %22 = trunc i64 %21 to i32
  ret i32 %22
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local ptr @create_map() alwaysinline #0 {
  %1 = alloca ptr, align 8
  %2 = alloca i32, align 4
  %3 = call noalias ptr @malloc(i64 noundef 800) #5
  store ptr %3, ptr %1, align 8
  store i32 0, ptr %2, align 4
  br label %4

4:                                                ; preds = %13, %0
  %5 = load i32, ptr %2, align 4
  %6 = icmp slt i32 %5, 100
  br i1 %6, label %7, label %16

7:                                                ; preds = %4
  %8 = load ptr, ptr %1, align 8
  %9 = getelementptr inbounds nuw %struct.HashMap, ptr %8, i32 0, i32 0
  %10 = load i32, ptr %2, align 4
  %11 = sext i32 %10 to i64
  %12 = getelementptr inbounds [100 x ptr], ptr %9, i64 0, i64 %11
  store ptr null, ptr %12, align 8
  br label %13

13:                                               ; preds = %7
  %14 = load i32, ptr %2, align 4
  %15 = add nsw i32 %14, 1
  store i32 %15, ptr %2, align 4
  br label %4, !llvm.loop !8

16:                                               ; preds = %4
  %17 = load ptr, ptr %1, align 8
  ret ptr %17
}

; Function Attrs: nounwind allocsize(0)
declare noalias ptr @malloc(i64 noundef) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @put(ptr noundef %0, ptr noundef %1, i32 noundef %2) alwaysinline #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store i32 %2, ptr %6, align 4
  %10 = load ptr, ptr %5, align 8
  %11 = call i32 @hash(ptr noundef %10)
  store i32 %11, ptr %7, align 4
  %12 = load ptr, ptr %4, align 8
  %13 = getelementptr inbounds nuw %struct.HashMap, ptr %12, i32 0, i32 0
  %14 = load i32, ptr %7, align 4
  %15 = zext i32 %14 to i64
  %16 = getelementptr inbounds nuw [100 x ptr], ptr %13, i64 0, i64 %15
  %17 = load ptr, ptr %16, align 8
  store ptr %17, ptr %8, align 8
  br label %18

18:                                               ; preds = %32, %3
  %19 = load ptr, ptr %8, align 8
  %20 = icmp ne ptr %19, null
  br i1 %20, label %21, label %36

21:                                               ; preds = %18
  %22 = load ptr, ptr %8, align 8
  %23 = getelementptr inbounds nuw %struct.Entry, ptr %22, i32 0, i32 0
  %24 = load ptr, ptr %23, align 8
  %25 = load ptr, ptr %5, align 8
  %26 = call i32 @strcmp(ptr noundef %24, ptr noundef %25) #6
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %28, label %32

28:                                               ; preds = %21
  %29 = load i32, ptr %6, align 4
  %30 = load ptr, ptr %8, align 8
  %31 = getelementptr inbounds nuw %struct.Entry, ptr %30, i32 0, i32 1
  store i32 %29, ptr %31, align 8
  br label %59

32:                                               ; preds = %21
  %33 = load ptr, ptr %8, align 8
  %34 = getelementptr inbounds nuw %struct.Entry, ptr %33, i32 0, i32 2
  %35 = load ptr, ptr %34, align 8
  store ptr %35, ptr %8, align 8
  br label %18, !llvm.loop !9

36:                                               ; preds = %18
  %37 = call noalias ptr @malloc(i64 noundef 24) #5
  store ptr %37, ptr %9, align 8
  %38 = load ptr, ptr %5, align 8
  %39 = call noalias ptr @strdup(ptr noundef %38) #7
  %40 = load ptr, ptr %9, align 8
  %41 = getelementptr inbounds nuw %struct.Entry, ptr %40, i32 0, i32 0
  store ptr %39, ptr %41, align 8
  %42 = load i32, ptr %6, align 4
  %43 = load ptr, ptr %9, align 8
  %44 = getelementptr inbounds nuw %struct.Entry, ptr %43, i32 0, i32 1
  store i32 %42, ptr %44, align 8
  %45 = load ptr, ptr %4, align 8
  %46 = getelementptr inbounds nuw %struct.HashMap, ptr %45, i32 0, i32 0
  %47 = load i32, ptr %7, align 4
  %48 = zext i32 %47 to i64
  %49 = getelementptr inbounds nuw [100 x ptr], ptr %46, i64 0, i64 %48
  %50 = load ptr, ptr %49, align 8
  %51 = load ptr, ptr %9, align 8
  %52 = getelementptr inbounds nuw %struct.Entry, ptr %51, i32 0, i32 2
  store ptr %50, ptr %52, align 8
  %53 = load ptr, ptr %9, align 8
  %54 = load ptr, ptr %4, align 8
  %55 = getelementptr inbounds nuw %struct.HashMap, ptr %54, i32 0, i32 0
  %56 = load i32, ptr %7, align 4
  %57 = zext i32 %56 to i64
  %58 = getelementptr inbounds nuw [100 x ptr], ptr %55, i64 0, i64 %57
  store ptr %53, ptr %58, align 8
  br label %59

59:                                               ; preds = %36, %28
  ret void
}

; Function Attrs: nounwind willreturn memory(read)
declare i32 @strcmp(ptr noundef, ptr noundef) #2

; Function Attrs: nounwind
declare noalias ptr @strdup(ptr noundef) #3

; Function Attrs: noinline nounwind optnone uwtable
define dso_local ptr @get(ptr noundef %0, ptr noundef %1) alwaysinline #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %8 = load ptr, ptr %5, align 8
  %9 = call i32 @hash(ptr noundef %8)
  store i32 %9, ptr %6, align 4
  %10 = load ptr, ptr %4, align 8
  %11 = getelementptr inbounds nuw %struct.HashMap, ptr %10, i32 0, i32 0
  %12 = load i32, ptr %6, align 4
  %13 = zext i32 %12 to i64
  %14 = getelementptr inbounds nuw [100 x ptr], ptr %11, i64 0, i64 %13
  %15 = load ptr, ptr %14, align 8
  store ptr %15, ptr %7, align 8
  br label %16

16:                                               ; preds = %29, %2
  %17 = load ptr, ptr %7, align 8
  %18 = icmp ne ptr %17, null
  br i1 %18, label %19, label %33

19:                                               ; preds = %16
  %20 = load ptr, ptr %7, align 8
  %21 = getelementptr inbounds nuw %struct.Entry, ptr %20, i32 0, i32 0
  %22 = load ptr, ptr %21, align 8
  %23 = load ptr, ptr %5, align 8
  %24 = call i32 @strcmp(ptr noundef %22, ptr noundef %23) #6
  %25 = icmp eq i32 %24, 0
  br i1 %25, label %26, label %29

26:                                               ; preds = %19
  %27 = load ptr, ptr %7, align 8
  %28 = getelementptr inbounds nuw %struct.Entry, ptr %27, i32 0, i32 1
  store ptr %28, ptr %3, align 8
  br label %34

29:                                               ; preds = %19
  %30 = load ptr, ptr %7, align 8
  %31 = getelementptr inbounds nuw %struct.Entry, ptr %30, i32 0, i32 2
  %32 = load ptr, ptr %31, align 8
  store ptr %32, ptr %7, align 8
  br label %16, !llvm.loop !10

33:                                               ; preds = %16
  store ptr null, ptr %3, align 8
  br label %34

34:                                               ; preds = %33, %26
  %35 = load ptr, ptr %3, align 8
  ret ptr %35
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @delete(ptr noundef %0, ptr noundef %1) alwaysinline #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %8 = load ptr, ptr %4, align 8
  %9 = call i32 @hash(ptr noundef %8)
  store i32 %9, ptr %5, align 4
  %10 = load ptr, ptr %3, align 8
  %11 = getelementptr inbounds nuw %struct.HashMap, ptr %10, i32 0, i32 0
  %12 = load i32, ptr %5, align 4
  %13 = zext i32 %12 to i64
  %14 = getelementptr inbounds nuw [100 x ptr], ptr %11, i64 0, i64 %13
  %15 = load ptr, ptr %14, align 8
  store ptr %15, ptr %6, align 8
  store ptr null, ptr %7, align 8
  br label %16

16:                                               ; preds = %49, %2
  %17 = load ptr, ptr %6, align 8
  %18 = icmp ne ptr %17, null
  br i1 %18, label %19, label %54

19:                                               ; preds = %16
  %20 = load ptr, ptr %6, align 8
  %21 = getelementptr inbounds nuw %struct.Entry, ptr %20, i32 0, i32 0
  %22 = load ptr, ptr %21, align 8
  %23 = load ptr, ptr %4, align 8
  %24 = call i32 @strcmp(ptr noundef %22, ptr noundef %23) #6
  %25 = icmp eq i32 %24, 0
  br i1 %25, label %26, label %49

26:                                               ; preds = %19
  %27 = load ptr, ptr %7, align 8
  %28 = icmp ne ptr %27, null
  br i1 %28, label %29, label %35

29:                                               ; preds = %26
  %30 = load ptr, ptr %6, align 8
  %31 = getelementptr inbounds nuw %struct.Entry, ptr %30, i32 0, i32 2
  %32 = load ptr, ptr %31, align 8
  %33 = load ptr, ptr %7, align 8
  %34 = getelementptr inbounds nuw %struct.Entry, ptr %33, i32 0, i32 2
  store ptr %32, ptr %34, align 8
  br label %44

35:                                               ; preds = %26
  %36 = load ptr, ptr %6, align 8
  %37 = getelementptr inbounds nuw %struct.Entry, ptr %36, i32 0, i32 2
  %38 = load ptr, ptr %37, align 8
  %39 = load ptr, ptr %3, align 8
  %40 = getelementptr inbounds nuw %struct.HashMap, ptr %39, i32 0, i32 0
  %41 = load i32, ptr %5, align 4
  %42 = zext i32 %41 to i64
  %43 = getelementptr inbounds nuw [100 x ptr], ptr %40, i64 0, i64 %42
  store ptr %38, ptr %43, align 8
  br label %44

44:                                               ; preds = %35, %29
  %45 = load ptr, ptr %6, align 8
  %46 = getelementptr inbounds nuw %struct.Entry, ptr %45, i32 0, i32 0
  %47 = load ptr, ptr %46, align 8
  call void @free(ptr noundef %47) #7
  %48 = load ptr, ptr %6, align 8
  call void @free(ptr noundef %48) #7
  br label %54

49:                                               ; preds = %19
  %50 = load ptr, ptr %6, align 8
  store ptr %50, ptr %7, align 8
  %51 = load ptr, ptr %6, align 8
  %52 = getelementptr inbounds nuw %struct.Entry, ptr %51, i32 0, i32 2
  %53 = load ptr, ptr %52, align 8
  store ptr %53, ptr %6, align 8
  br label %16, !llvm.loop !11

54:                                               ; preds = %44, %16
  ret void
}

; Function Attrs: nounwind
declare void @free(ptr noundef) #3

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @free_map(ptr noundef %0) alwaysinline #0 {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  store i32 0, ptr %3, align 4
  br label %6

6:                                                ; preds = %29, %1
  %7 = load i32, ptr %3, align 4
  %8 = icmp slt i32 %7, 100
  br i1 %8, label %9, label %32

9:                                                ; preds = %6
  %10 = load ptr, ptr %2, align 8
  %11 = getelementptr inbounds nuw %struct.HashMap, ptr %10, i32 0, i32 0
  %12 = load i32, ptr %3, align 4
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds [100 x ptr], ptr %11, i64 0, i64 %13
  %15 = load ptr, ptr %14, align 8
  store ptr %15, ptr %4, align 8
  br label %16

16:                                               ; preds = %19, %9
  %17 = load ptr, ptr %4, align 8
  %18 = icmp ne ptr %17, null
  br i1 %18, label %19, label %28

19:                                               ; preds = %16
  %20 = load ptr, ptr %4, align 8
  store ptr %20, ptr %5, align 8
  %21 = load ptr, ptr %4, align 8
  %22 = getelementptr inbounds nuw %struct.Entry, ptr %21, i32 0, i32 2
  %23 = load ptr, ptr %22, align 8
  store ptr %23, ptr %4, align 8
  %24 = load ptr, ptr %5, align 8
  %25 = getelementptr inbounds nuw %struct.Entry, ptr %24, i32 0, i32 0
  %26 = load ptr, ptr %25, align 8
  call void @free(ptr noundef %26) #7
  %27 = load ptr, ptr %5, align 8
  call void @free(ptr noundef %27) #7
  br label %16, !llvm.loop !12

28:                                               ; preds = %16
  br label %29

29:                                               ; preds = %28
  %30 = load i32, ptr %3, align 4
  %31 = add nsw i32 %30, 1
  store i32 %31, ptr %3, align 4
  br label %6, !llvm.loop !13

32:                                               ; preds = %6
  %33 = load ptr, ptr %2, align 8
  call void @free(ptr noundef %33) #7
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  %3 = call i32 (ptr, ...) @printf(ptr noundef @.str)
  %4 = call i32 (...) @test_dict_operations()
  store i32 %4, ptr %2, align 4
  %5 = load i32, ptr %2, align 4
  %6 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %5)
  ret i32 0
}

declare i32 @printf(ptr noundef, ...) #4

declare i32 @test_dict_operations(...) #4

attributes #0 = { nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind willreturn memory(read) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nounwind allocsize(0) }
attributes #6 = { nounwind willreturn memory(read) }
attributes #7 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"Ubuntu clang version 20.1.7 (++20250612073421+199e02a36433-1~exp1~20250612193439.130)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
