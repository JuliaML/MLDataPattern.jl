@testset "RandomObs (and overlaps with BalancedObs)" begin
    for TIter in (RandomObs, BalancedObs)
        @test TIter <: LearnBase.DataIterator
        @test TIter <: LearnBase.ObsIterator
        @test TIter <: LearnBase.AbstractDataIterator
        @test TIter <: LearnBase.AbstractObsIterator
    end
    @test_reference "references/RandomObs1.txt" RandomObs(X) by=matrix_compat_isequal
    @test_reference "references/RandomObs2.txt" RandomObs(X, 10) by=matrix_compat_isequal
    @test_reference "references/BalancedObs1.txt" BalancedObs(X) by=matrix_compat_isequal
    @test_reference "references/BalancedObs2.txt" BalancedObs(X, 10) by=matrix_compat_isequal

    for TIter in (RandomObs, BalancedObs)
        @testset "constructor for $TIter" begin
            for var in (vars..., tuples..., Xs, ys)
                A = @inferred TIter(var)
                @test_throws MethodError collect(A)
                @test typeof(A) <: TIter
                @test @inferred(nobs(A)) == 150
                @test_throws MethodError length(A)
                @test typeof(@inferred(first(A))) == typeof(datasubset(var, 1))
                A = @inferred TIter(var, 100)
                @test typeof(A) <: TIter
                @test @inferred(nobs(A)) == 150
                @test @inferred(length(A)) == 100
                @test typeof(@inferred(first(A))) == typeof(datasubset(var, 1))
                @test eltype(@inferred(collect(A))) == typeof(datasubset(var, 1))
                B = TIter(var, count = 100)
                @test typeof(B) == typeof(A)
                @test A.count == B.count
            end
            for var in (X,XX,XXX)
                A = @inferred TIter(var, 100)
                @test size(getobs(first(A))) == size(getobs(datasubset(var, 1)))
            end
        end
    end

    for TIter in (RandomObs, BalancedObs)
        @testset "DataSubset: $TIter" begin
            for raw in (vars..., tuples..., Xs, ys)
                var = DataSubset(raw, 1:90)
                A = @inferred TIter(var)
                @test_throws MethodError collect(A)
                @test typeof(A) <: TIter
                @test @inferred(nobs(A)) == 90
                @test_throws MethodError length(A)
                @test_throws MethodError getobs(A)
                @test_throws MethodError getobs(A, 1)
                @test typeof(@inferred(first(A))) == typeof(datasubset(var, 1))
                A = @inferred TIter(var, 100)
                @test typeof(A) <: TIter
                @test @inferred(nobs(A)) == 90
                @test @inferred(length(A)) == 100
                @test_throws MethodError getobs(A, 1)
                @test typeof(@inferred(first(A))) == typeof(datasubset(var, 1))
                @test eltype(@inferred(collect(A))) == typeof(datasubset(var, 1))
                @test typeof(@inferred(getobs(A))) == typeof(getobs.(collect(A)))
            end
        end
    end

    for TIter in (RandomObs, BalancedObs)
        @testset "various obsdim: $TIter" begin
            A = @inferred TIter(X', ObsDim.First())
            B = TIter(X', obsdim = 1)
            @test A.data == B.data
            @test A.obsdim == B.obsdim
            @test A.count == B.count
            A = @inferred TIter(X', 10, ObsDim.First())
            B = TIter(X', 10, obsdim = 1)
            C = TIter(X', count = 10, obsdim = 1)
            @test A.data == B.data == C.data
            @test A.obsdim == B.obsdim == C.obsdim
            @test A.count == B.count == C.count
            @test typeof(@inferred(first(A))) == typeof(datasubset(X', 1, ObsDim.First()))
            @test eltype(@inferred(collect(A))) == typeof(datasubset(X', 1, ObsDim.First()))
            @test size(getobs(first(A))) == size(getobs(datasubset(X', 1, ObsDim.First())))
            A = @inferred TIter((X',X'), 10, ObsDim.First())
            @test typeof(@inferred(first(A))) == typeof(datasubset((X',X'), 1, ObsDim.First()))
            @test eltype(@inferred(collect(A))) == typeof(datasubset((X',X'), 1, ObsDim.First()))
            A = @inferred TIter((X',X), 10, (ObsDim.First(),ObsDim.Last()))
            @test typeof(@inferred(first(A))) == typeof(datasubset((X',X), 1, (ObsDim.First(),ObsDim.Last())))
            @test eltype(@inferred(collect(A))) == typeof(datasubset((X',X), 1, (ObsDim.First(),ObsDim.Last())))
            A = @inferred TIter(datasubset(X', ObsDim.First()))
        end
    end

    for TIter in (RandomObs, BalancedObs)
        @testset "custom types: TIter" begin
            @test_throws MethodError TIter(EmptyType())
            @test_throws MethodError TIter(EmptyType(), 10)
            @test_throws MethodError TIter(EmptyType(), obsdim=1)
            @test_throws MethodError TIter(EmptyType(), 10, obsdim=1)
            A = @inferred TIter(CustomType())
            @test_throws MethodError collect(A)
            @test typeof(A) <: TIter
            @test @inferred(nobs(A)) == 100
            @test_throws MethodError length(A)
            @test typeof(@inferred(first(A))) == typeof(datasubset(CustomType(), 1))
            A = @inferred TIter(CustomType(), 500)
            @test typeof(A) <: TIter
            @test @inferred(nobs(A)) == 100
            @test @inferred(length(A)) == 500
            @test typeof(@inferred(first(A))) == typeof(datasubset(CustomType(), 1))
            @test eltype(@inferred(collect(A))) == typeof(datasubset(CustomType(), 1))
            @test all(getobs.(collect(A)) .<= 100)
        end
    end
end

@testset "BalancedObs" begin
    @testset "check balancing" begin
        A = @inferred BalancedObs((1:10,[:a,:b,:a,:a,:a,:a,:a,:a,:b,:a]), 100)
        @test A.labelmap == Dict(:a => [1,3,4,5,6,7,8,10], :b => [2,9])
        val = @inferred(map(getobs, A))
        freq = labelfreq(getindex.(val, 2))
        @test freq[:a] > 35
        @test freq[:b] > 35

        A = @inferred BalancedObs(i->i>1, 1:10, 100)
        @test A.labelmap == Dict(false => [1], true => [2,3,4,5,6,7,8,9,10])
        val = @inferred(map(getobs, A))
        @test sum(val .> 1) > 35
    end
end

@testset "RandomBatches" begin
    @test RandomBatches <: LearnBase.DataIterator
    @test RandomBatches <: LearnBase.BatchIterator
    @test RandomBatches <: LearnBase.AbstractDataIterator
    @test RandomBatches <: LearnBase.AbstractBatchIterator
    @test_reference "references/RandomBatches1.txt" RandomBatches(X) by=matrix_compat_isequal
    @test_reference "references/RandomBatches2.txt" RandomBatches(X,10,10) by=matrix_compat_isequal

    @testset "constructor" begin
        A = @inferred RandomBatches(rand(2,5), 10)
        @test typeof(A) <: RandomBatches
        @test @inferred(nobs(A)) == 5
        @test @inferred(batchsize(A)) == 10
        @test typeof(@inferred(first(A))) == typeof(datasubset(rand(2,5), rand(1:5, 10)))

        for var in (vars..., tuples..., Xs, ys)
            A = @inferred RandomBatches(var)
            @test_throws MethodError collect(A)
            @test typeof(A) <: RandomBatches
            @test @inferred(nobs(A)) == 150
            @test @inferred(batchsize(A)) == 30
            @test_throws MethodError length(A)
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, collect(1:30)))
            A = @inferred RandomBatches(var, 50)
            @test_throws MethodError collect(A)
            @test typeof(A) <: RandomBatches
            @test @inferred(nobs(A)) == 150
            @test @inferred(batchsize(A)) == 50
            @test_throws MethodError length(A)
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, collect(1:50)))
            B = RandomBatches(var, size = 50)
            @test typeof(A) == typeof(B)
            @test A.data == B.data
            @test A.count == B.count
            @test A.size == B.size
            @test A.obsdim == B.obsdim
            A = @inferred RandomBatches(var, 40, 100)
            @test typeof(A) <: RandomBatches
            @test @inferred(nobs(A)) == 150
            @test @inferred(batchsize(A)) == 40
            @test @inferred(length(A)) == 100
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, collect(1:40)))
            @test eltype(@inferred(collect(A))) == typeof(datasubset(var, collect(1:40)))
            B = RandomBatches(var, size = 40, count = 100)
            @test typeof(A) == typeof(B)
            @test A.data == B.data
            @test A.count == B.count
            @test A.size == B.size
            @test A.obsdim == B.obsdim
        end
        for var in (X,XX,XXX)
            A = @inferred RandomBatches(var)
            @test size(getobs(first(A))) == size(getobs(datasubset(var, collect(1:30))))
            A = @inferred RandomBatches(var, 50)
            @test size(getobs(first(A))) == size(getobs(datasubset(var, collect(1:50))))
            A = @inferred RandomBatches(var, 40, 100)
            @test size(getobs(first(A))) == size(getobs(datasubset(var, collect(1:40))))
        end
    end

    @testset "DataSubset" begin
        for raw in (vars..., tuples..., Xs, ys)
            var = DataSubset(raw)
            A = @inferred RandomBatches(var)
            @test_throws MethodError collect(A)
            @test typeof(A) <: RandomBatches
            @test @inferred(nobs(A)) == 150
            @test @inferred(batchsize(A)) == 30
            @test_throws MethodError length(A)
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, collect(1:30)))
            A = @inferred RandomBatches(var, 50)
            @test_throws MethodError collect(A)
            @test typeof(A) <: RandomBatches
            @test @inferred(nobs(A)) == 150
            @test @inferred(batchsize(A)) == 50
            @test_throws MethodError length(A)
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, collect(1:50)))
            A = @inferred RandomBatches(var, 40, 100)
            @test typeof(A) <: RandomBatches
            @test @inferred(nobs(A)) == 150
            @test @inferred(batchsize(A)) == 40
            @test @inferred(length(A)) == 100
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, collect(1:40)))
            @test eltype(@inferred(collect(A))) == typeof(datasubset(var, collect(1:40)))
        end
    end

    @testset "various obsdim" begin
        A = @inferred RandomBatches(X', 10, ObsDim.First())
        B = RandomBatches(X', 10, obsdim = 1)
        @test A.data == B.data
        @test A.obsdim == B.obsdim
        @test A.count == B.count
        @test A.size == B.size == batchsize(A)
        A = @inferred RandomBatches(X', 10, 100, ObsDim.First())
        B = RandomBatches(X', 10, 100, obsdim = 1)
        C = RandomBatches(X', size = 10, count = 100, obsdim = 1)
        @test A.data == B.data == C.data
        @test A.obsdim == B.obsdim == C.obsdim
        @test A.count == B.count == C.count
        @test A.size == B.size == C.size == batchsize(A)
        @test typeof(@inferred(first(A))) == typeof(datasubset(X', collect(1:10), ObsDim.First()))
        @test eltype(@inferred(collect(A))) == typeof(datasubset(X', collect(1:10), ObsDim.First()))
        @test size(getobs(first(A))) == size(getobs(datasubset(X', collect(1:10), ObsDim.First())))
        A = @inferred RandomBatches(DataSubset(X', ObsDim.First()), 10, 100)
        @test size(getobs(first(A))) == size(getobs(datasubset(X', collect(1:10), ObsDim.First())))
    end

    @testset "custom types" begin
        @test_throws MethodError RandomBatches(EmptyType())
        @test_throws MethodError RandomBatches(EmptyType(), 10)
        @test_throws MethodError RandomBatches(EmptyType(), 10, 100)
        @test_throws MethodError RandomBatches(EmptyType(), obsdim=1)
        @test_throws MethodError RandomBatches(EmptyType(), 10, obsdim=1)
        A = @inferred RandomBatches(CustomType(), 20)
        @test_throws MethodError collect(A)
        @test typeof(A) <: RandomBatches
        @test @inferred(nobs(A)) == 100
        @test @inferred(batchsize(A)) == 20
        @test_throws MethodError length(A)
        @test typeof(@inferred(first(A))) == typeof(datasubset(CustomType(), collect(1:20)))
        A = @inferred RandomBatches(CustomType(), 20, 500)
        @test typeof(A) <: RandomBatches
        @test @inferred(nobs(A)) == 100
        @test @inferred(length(A)) == 500
        @test @inferred(batchsize(A)) == 20
        @test typeof(@inferred(first(A))) == typeof(datasubset(CustomType(), collect(1:20)))
        @test eltype(@inferred(collect(A))) == typeof(datasubset(CustomType(), collect(1:20)))
    end
end

@testset "BufferGetObs" begin
    @testset "ObsView" begin
        A = BufferGetObs(ObsView(X))
        @test_reference "references/BufferGetObs1.txt" A by=matrix_compat_isequal

        @test size(A.buffer) == (4,)
        @test typeof(A.buffer) <: Array{Float64,1}

        for var in (X,Y,XX,XXX,(X,Y),(XX,X),(XXX,XX,X))
            A = @inferred BufferGetObs(ObsView(var))
            @test typeof(A.buffer) == typeof(getobs(var,1))
            for (i,a) in enumerate(A)
                @test a === A.buffer
                @test a == getobs(var,i)
            end
        end

        # preallocating buffer
        for var in (X,Y,XX,XXX)
            buffer = similar(getobs(var,1))
            A = @inferred BufferGetObs(ObsView(var), buffer)
            for (i,a) in enumerate(A)
                @test a === A.buffer
                @test a === buffer
                @test a == getobs(var,i)
            end
        end
        # preallocating buffer for tuple
        for var in ((X,X),(XX,X),(X,Y))
            buffer = (similar(getobs(var[1],1)),similar(getobs(var[2],1)))
            A = @inferred BufferGetObs(ObsView(var), buffer)
            for (i,a) in enumerate(A)
                @test a === A.buffer
                @test a === buffer
                @test a == getobs(var,i)
            end
        end

        for var in (vars..., tuples..., Xs, ys, CustomType())
            A = @inferred BufferGetObs(ObsView(var))
            @test typeof(A.buffer) == typeof(getobs(var,1))
            for (i,a) in enumerate(A)
                @test a == getobs(var,i)
            end
        end
    end

    @testset "BatchView" begin
        A = BufferGetObs(BatchView(X, 15))
        @test size(A.buffer) == (4,15)
        @test typeof(A.buffer) <: Array{Float64,2}

        for var in (X,Y,XX,XXX,(X,Y),(XX,X),(XXX,XX,X))
            A = @inferred BufferGetObs(BatchView(var,15))
            @test typeof(A.buffer) == typeof(getobs(var,collect(1:15)))
            for (i,a) in enumerate(A)
                @test a === A.buffer
                @test a == BatchView(var,15)[i]
            end
        end
        # preallocating buffer
        for var in (X,Y,XX,XXX)
            buffer = similar(getobs(var,1:15))
            A = @inferred BufferGetObs(BatchView(var,15), buffer)
            for (i,a) in enumerate(A)
                @test a === A.buffer
                @test a === buffer
                @test a == BatchView(var,15)[i]
            end
        end
        # preallocating buffer for tuple
        for var in ((X,X),(XX,X),(X,Y))
            buffer = (similar(getobs(var[1],1:15)),similar(getobs(var[2],1:15)))
            A = @inferred BufferGetObs(BatchView(var, 15), buffer)
            for (i,a) in enumerate(A)
                @test a === A.buffer
                @test a === buffer
                @test a == BatchView(var,15)[i]
            end
        end

        for var in (vars..., tuples..., Xs, ys, CustomType())
            A = @inferred BufferGetObs(BatchView(var,10))
            @test typeof(A.buffer) == typeof(getobs(var,1:10))
            for (i,a) in enumerate(A)
                @test a == getobs(BatchView(var,10)[i])
            end
        end
    end

    @testset "RandomObs" begin
        A = BufferGetObs(RandomObs(X, 10))
        @test size(A.buffer) == (4,)
        @test typeof(A.buffer) <: Array{Float64,1}
        @test length(A) == 10
        @test Base.IteratorSize(A) == Base.HasLength()
        A = BufferGetObs(RandomObs(X))
        @test size(A.buffer) == (4,)
        @test typeof(A.buffer) <: Array{Float64,1}
        @test_throws MethodError length(A)
        @test Base.IteratorSize(A) == Base.IsInfinite()

        for var in (X,Y,XX,XXX,(X,Y),(XX,X),(XXX,XX,X))
            A = @inferred BufferGetObs(RandomObs(var,10))
            @test typeof(A.buffer) == typeof(getobs(var,1))
            for (i,a) in enumerate(A)
                @test a === A.buffer
                @test i <= 10
            end
        end
        for var in (vars..., tuples..., Xs, ys, CustomType())
            A = @inferred BufferGetObs(RandomObs(var, 10))
            @test typeof(A.buffer) == typeof(getobs(var,1))
            for (i,a) in enumerate(A)
                @test i <= 10
            end
        end
    end

    @testset "RandomBatches" begin
        A = BufferGetObs(RandomBatches(X, 15, 10))
        @test size(A.buffer) == (4,15)
        @test typeof(A.buffer) <: Array{Float64,2}
        @test length(A) == 10
        @test Base.IteratorSize(A) == Base.HasLength()
        A = BufferGetObs(RandomBatches(X, 15))
        @test size(A.buffer) == (4,15)
        @test typeof(A.buffer) <: Array{Float64,2}
        @test_throws MethodError length(A)
        @test Base.IteratorSize(A) == Base.IsInfinite()

        for var in (X,Y,XX,XXX,(X,Y),(XX,X),(XXX,XX,X))
            A = @inferred BufferGetObs(RandomBatches(var,15,10))
            @test typeof(A.buffer) == typeof(getobs(var,collect(1:15)))
            for (i,a) in enumerate(A)
                @test a === A.buffer
                @test i <= 10
            end
        end
        for var in (vars..., tuples..., Xs, ys, CustomType())
            A = @inferred BufferGetObs(RandomBatches(var,10,10))
            @test typeof(A.buffer) == typeof(getobs(var,1:10))
            for (i,a) in enumerate(A)
                @test i <= 10
            end
        end
    end
end

@testset "eachobs" begin
    A = @inferred eachobs(X)
    @inferred first(A)
    @test typeof(A) <: BufferGetObs
    @test typeof(A.iter) <: ObsView
    @test A.iter.obsdim == ObsDim.Last()
    @test A.iter.data == X
    A = @inferred eachobs(X, ObsDim.First())
    @inferred first(A)
    @test typeof(A) <: BufferGetObs
    @test typeof(A.iter) <: ObsView
    @test A.iter.obsdim == ObsDim.First()
    @test A.iter.data == X
    A = eachobs(X, obsdim = 1)
    @inferred first(A)
    @test typeof(A) <: BufferGetObs
    @test typeof(A.iter) <: ObsView
    @test A.iter.obsdim == ObsDim.First()
    @test A.iter.data == X
end

@testset "eachbatch" begin
    A = @inferred eachbatch(X)
    @inferred first(A)
    @test typeof(A) <: BufferGetObs
    @test typeof(A.iter) <: BatchView
    @test A.iter.obsdim == ObsDim.Last()
    @test A.iter.data == X
    @test size(A.buffer) == (4,30)
    @test length(A) == 5
    A = @inferred eachbatch(X', ObsDim.First())
    @inferred first(A)
    @test typeof(A) <: BufferGetObs
    @test typeof(A.iter) <: BatchView
    @test A.iter.obsdim == ObsDim.First()
    @test A.iter.data == X'
    @test size(A.buffer) == (30,4)
    @test length(A) == 5
    A = eachbatch(X', obsdim=1)
    @inferred first(A)
    @test typeof(A) <: BufferGetObs
    @test typeof(A.iter) <: BatchView
    @test A.iter.obsdim == ObsDim.First()
    @test A.iter.data == X'
    @test size(A.buffer) == (30,4)
    @test length(A) == 5
    A = @inferred eachbatch(X, 10)
    @test_throws ArgumentError eachbatch(X, size = 10, maxsize = 11)
    @test length(A) == length(eachbatch(X, maxsize = 11))
    @inferred first(A)
    @test typeof(A) <: BufferGetObs
    @test typeof(A.iter) <: BatchView
    @test A.iter.obsdim == ObsDim.Last()
    @test A.iter.data == X
    @test size(A.buffer) == (4,10)
    @test length(A) == 15
    A = @inferred eachbatch(X', 10, ObsDim.First())
    @inferred first(A)
    @test typeof(A) <: BufferGetObs
    @test typeof(A.iter) <: BatchView
    @test A.iter.obsdim == ObsDim.First()
    @test A.iter.data == X'
    @test size(A.buffer) == (10,4)
    @test length(A) == 15
    A = eachbatch(X', count = 15, obsdim = 1)
    @inferred first(A)
    @test typeof(A) <: BufferGetObs
    @test typeof(A.iter) <: BatchView
    @test A.iter.obsdim == ObsDim.First()
    @test A.iter.data == X'
    @test size(A.buffer) == (10,4)
    @test length(A) == 15
end
