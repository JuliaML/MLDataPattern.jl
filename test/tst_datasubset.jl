@testset "DataSubset constructor" begin
    @test_throws DimensionMismatch DataSubset((rand(2,10),rand(9)))
    @test_throws DimensionMismatch DataSubset((rand(2,10),rand(9)),1:2)
    @test_throws DimensionMismatch DataSubset((rand(2,10),rand(4,9,10),rand(9)))

    @testset "bounds check" begin
        for var in (vars..., tuples..., CustomType())
            @test_throws BoundsError getobs(DataSubset(var, -1:100), 1)
            @test_throws BoundsError getobs(DataSubset(var, 1:151), 151)
            @test_throws BoundsError getobs(DataSubset(var, [1, 10, 0, 3]), 3)
            @test_throws BoundsError getobs(DataSubset(var, [1, 10, -10, 3]), 3)
            # todo: finish this
            @test_throws BoundsError getobs(DataSubset(var, [1, 10, 180, 3]), 3)
        end
    end

    # @testset "Tuple unrolling" begin
    #     @test_throws DimensionMismatch DataSubset((X,X), 1:150; obsdim = (2, 2, 2))
    #     @test_throws DimensionMismatch DataSubset((X,X), 1:150; obsdim = (2,))
    #     @test_throws DimensionMismatch DataSubset((X,X); obsdim = (2, 2, 2))
    #     @test_throws DimensionMismatch DataSubset((X,X); obsdim = (2,))
    #     @test typeof(@inferred(DataSubset((X,X)))) <: Tuple
    #     @test eltype(@inferred(DataSubset((X,X)))) <: DataSubset
    #     @test typeof(@inferred(DataSubset((X,X); obsdim = 2))) <: Tuple
    #     @test eltype(@inferred(DataSubset((X,X); obsdim = 2))) <: DataSubset
    #     @test typeof(@inferred(DataSubset((X,X); obsdim = (2, 2)))) <: Tuple
    #     @test eltype(@inferred(DataSubset((X,X); obsdim = (2, 2)))) <: DataSubset
    #     @test typeof(@inferred(DataSubset((X,X), 1:150; obsdim = (2, 2)))) <: Tuple
    #     @test eltype(@inferred(DataSubset((X,X), 1:150; obsdim = (2, 2)))) <: DataSubset
    #     D1 = @inferred(DataSubset((X',X); obsdim = (1, 2)))
    #     D2 = @inferred(DataSubset((X',X), 1:150; obsdim = (1, 2)))
    #     D3 = DataSubset((X',X); obsdim = (1, 2))
    #     D4 = DataSubset((X',X), 1:150; obsdim = (1, 2))
    #     for (s1,s2) in (D1,D2,D3,D4)
    #         @test typeof(datasubset(s1,2:10)) <: DataSubset
    #         @test @inferred(datasubset(s1,2:10)) == @inferred(s1[2:10])
    #         @test @inferred(datasubset(s1,2:10)) == @inferred(DataSubset(s1,2:10))
    #         @test s1.obsdim == ObsDim.First()
    #         @test s2.obsdim == ObsDim.Last()
    #         @test getobs(s1,2) == getobs(s2,2)
    #         @test getobs(s1,9:10) == getobs(s2,9:10)'
    #         @test getobs((s1,s2),9:10) == (getobs(s1,9:10),getobs(s2,9:10))
    #         @test nobs(s1) == nobs(s2) == 150
    #     end
    # end

    # @testset "Array, SubArray, SparseArray" begin
    #     @test nobs(DataSubset(X; obsdim = 1)) == 4
    #     @test nobs(DataSubset(X, 1:3); obsdim = 1) == 3
    #     @test_reference "references/DataSubset1.txt" DataSubset(X, Int64(1):Int64(nobs(X))) by=matrix_compat_isequal
    #     @test_reference "references/DataSubset2.txt" @io2str(showcompact(::IO, DataSubset(X))) by=matrix_compat_isequal
    #     for var in (Xs, ys, vars...)
    #         subset = @inferred(DataSubset(var))
    #         @test subset.data === var
    #         @test subset.indices === 1:150
    #         @test typeof(subset) <: DataSubset
    #         @test @inferred(nobs(subset)) === nobs(var)
    #         @test @inferred(getobs(subset)) == getobs(var)
    #         @test @inferred(DataSubset(subset)) === subset
    #         @test @inferred(DataSubset(subset, 1:150)) === subset
    #         @test subset[end] == DataSubset(var, 150)
    #         @test @inferred(subset[150]) == DataSubset(var, 150)
    #         @test @inferred(subset[20:25]) == DataSubset(var, 20:25)
    #         for idx in (1:100, [1,10,150,3], [2])
    #             @test DataSubset(var)[idx] == DataSubset(var, idx)
    #             @test DataSubset(var)[idx] == DataSubset(var, collect(idx))
    #             subset = @inferred(DataSubset(var, idx))
    #             @test typeof(subset) <: DataSubset{typeof(var), typeof(idx)}
    #             @test subset.data === var
    #             @test subset.indices === idx
    #             @test subset.obsdim === ObsDim.Last()
    #             @test @inferred(nobs(subset)) === length(idx)
    #             @test @inferred(getobs(subset)) == getobs(var, idx)
    #             @test @inferred(DataSubset(subset)) === subset
    #             @test @inferred(subset[1]) == DataSubset(var, idx[1])
    #             if typeof(idx) <: AbstractRange
    #                 @test typeof(@inferred(subset[1:1])) == typeof(DataSubset(var, idx[1:1]))
    #                 @test nobs(subset[1:1]) == nobs(DataSubset(var, idx[1:1]))
    #             else
    #                 @test typeof(@inferred(subset[1:1])) == typeof(DataSubset(var, view(idx, 1:1)))
    #                 @test nobs(subset[1:1]) == nobs(DataSubset(var, view(idx, 1:1)))
    #             end
    #         end
    #     end
    # end

    # @testset "custom types" begin
    #     @test_throws MethodError DataSubset(EmptyType())
    #     @test_throws MethodError DataSubset(EmptyType(), 1:10)
    #     @test_throws MethodError DataSubset(EmptyType(), 1:10, ObsDim.First())
    #     @test_throws MethodError DataSubset(CustomType(), obsdim=1)
    #     @test_throws MethodError DataSubset(CustomType(), obsdim=:last)
    #     @test_throws MethodError DataSubset(CustomType(), 2:10, obsdim=1)
    #     @test_throws MethodError DataSubset(CustomType(), 2:10, obsdim=:last)
    #     @test_throws BoundsError getobs(DataSubset(CustomType(), 11:20), 11)
    #     @test typeof(@inferred(DataSubset(CustomType()))) <: DataSubset
    #     @test nobs(DataSubset(CustomType())) === 100
    #     @test nobs(DataSubset(CustomType(), 11:20)) === 10
    #     @test getobs(DataSubset(CustomType())) == collect(1:100)
    #     @test getobs(DataSubset(CustomType(),11:20),10) == 20
    #     @test getobs(DataSubset(CustomType(),11:20),[3,5]) == [13,15]
    # end
end

# @testset "DataSubset getindex and getobs" begin
#     @testset "Matrix and SubArray{T,2}" begin
#         for var in (X, Xv)
#             subset = @inferred(DataSubset(var, 101:150))
#             @test typeof(@inferred(getobs(subset))) <: Array{Float64,2}
#             @test @inferred(nobs(subset)) == length(subset) == 50
#             @test @inferred(subset[10:20]) == DataSubset(X, 110:120)
#             @test @inferred(subset[11:21]) != DataSubset(X, 110:120)
#             @test @inferred(getobs(subset, 10:20)) == X[:, 110:120]
#             @test @inferred(getobs(subset, [11,10,14])) == X[:, [111,110,114]]
#             @test typeof(subset[10:20]) <: DataSubset
#             @test @inferred(subset[collect(10:20)]) == DataSubset(X, collect(110:120))
#             @test typeof(subset[collect(10:20)]) <: DataSubset
#             @test @inferred(getobs(subset)) == getobs(subset[1:end]) == X[:, 101:150]
#         end
#     end

#     @testset "Vector and SubArray{T,1}" begin
#         for var in (y, yv)
#             subset = @inferred(DataSubset(var, 101:150))
#             @test typeof(getobs(subset)) <: Array{String,1}
#             @test @inferred(nobs(subset)) == length(subset) == 50
#             @test @inferred(subset[10:20]) == DataSubset(y, 110:120)
#             @test @inferred(getobs(subset, 10:20)) == y[110:120]
#             @test @inferred(getobs(subset, [11,10,14])) == y[[111,110,114]]
#             @test typeof(subset[10:20]) <: DataSubset
#             @test @inferred(subset[collect(10:20)]) == DataSubset(y, collect(110:120))
#             @test typeof(subset[collect(10:20)]) <: DataSubset
#             @test @inferred(getobs(subset)) == getobs(subset[1:end]) == y[101:150]
#         end
#     end

#     @testset "2-Tuple of Matrix, Vector, or SubArray"  begin
#         for v1 in (X, Xv), v2 in (y, yv)
#             subset = @inferred(DataSubset((v1,v2), 101:150))
#             @test typeof(getobs(subset)) <: Tuple{Array{Float64,2},Array{String,1}}
#             @test @inferred(nobs(subset)) == nobs(subset[1]) == nobs(subset[2]) == 50
#             @test @inferred(subset[1][10:20]) == DataSubset(X, 110:120)
#             @test @inferred(subset[2][10:20]) == DataSubset(y, 110:120)
#             @test @inferred(getobs(subset)) == (X[:, 101:150], y[101:150])
#         end
#     end

#     @testset "2-Tuple of SparseArray"  begin
#         subset = @inferred(DataSubset((Xs,ys), 101:150))
#         @test typeof(subset) <: Tuple
#         @test typeof(subset[1]) <: DataSubset
#         @test typeof(subset[2]) <: DataSubset
#         @test typeof(@inferred(getobs(subset))) <: Tuple
#         @test typeof(getobs(subset)[1]) <: SparseMatrixCSC
#         @test typeof(getobs(subset)[2]) <: SparseVector
#         @test @inferred(nobs(subset)) == nobs(subset[1]) == nobs(subset[2]) == 50
#         @test @inferred(getobs(subset[1][10:20])) == getindex(Xs, :, 110:120)
#         @test @inferred(getobs(subset[2][10:20])) == getindex(ys, 110:120)
#         @test @inferred(getobs(subset)) == (getindex(Xs, :, 101:150), getindex(ys, 101:150))
#     end
# end

println("<HEARTBEAT>")

# @testset "datasubset" begin
#     @testset "Array and SubArray" begin
#         @test @inferred(datasubset(X)) == Xv
#         @test @inferred(datasubset(X, ObsDim.Last())) == Xv
#         @test @inferred(datasubset(X, ObsDim.Last())) == Xv
#         @test typeof(datasubset(X)) <: SubArray
#         @test @inferred(datasubset(Xv)) === Xv
#         @test @inferred(datasubset(XX)) == XX
#         @test @inferred(datasubset(XXX)) == XXX
#         @test typeof(datasubset(XXX)) <: SubArray
#         @test @inferred(datasubset(y)) == y
#         @test typeof(datasubset(y)) <: SubArray
#         @test @inferred(datasubset(yv)) === yv
#         for i in (2, 1:150, 2:10, [2,5,7], [2,1])
#             @test @inferred(datasubset(X,i))   === view(X,:,i)
#             @test @inferred(datasubset(Xv,i))  === view(X,:,i)
#             @test @inferred(datasubset(Xv,i))  === view(Xv,:,i)
#             @test @inferred(datasubset(XX,i))  === view(XX,:,:,i)
#             @test @inferred(datasubset(XXX,i)) === view(XXX,:,:,:,i)
#             @test @inferred(datasubset(y,i))   === view(y,i)
#             @test @inferred(datasubset(yv,i))  === view(y,i)
#             @test @inferred(datasubset(yv,i))  === view(yv,i)
#             @test @inferred(datasubset(Y,i))   === view(Y,:,i)
#         end
#     end

#     @testset "Tuple of Array and Subarray" begin
#         @test_throws DimensionMismatch datasubset((X,X), 1:150, (ObsDim.Last(), ObsDim.Last(), ObsDim.Last()))
#         @test_throws DimensionMismatch datasubset((X,X), 1:150, (ObsDim.Last(),))
#         @test_throws DimensionMismatch datasubset((X,X), (ObsDim.Last(), ObsDim.Last(), ObsDim.Last()))
#         @test_throws DimensionMismatch datasubset((X,X), (ObsDim.Last(),))
#         @test @inferred(datasubset((X,y),ObsDim.Last())) == (X,y)
#         @test @inferred(datasubset((X,y),(ObsDim.Last(),ObsDim.Last()))) == (X,y)
#         @test @inferred(datasubset((X,y)))   == (X,y)
#         @test @inferred(datasubset((X,yv)))  == (X,yv)
#         @test @inferred(datasubset((X,yv)))  === (view(X,:,1:150),yv)
#         @test @inferred(datasubset((Xv,y)))  == (Xv,y)
#         @test @inferred(datasubset((Xv,y)))  === (Xv,view(y,1:150))
#         @test @inferred(datasubset((Xv,yv))) === (Xv,yv)
#         @test @inferred(datasubset((X,Y)))   == (X,Y)
#         @test @inferred(datasubset((XX,X,y))) == (XX,X,y)
#         for i in (1:150, 2:10, [2,5,7], [2,1])
#             @test @inferred(datasubset((X,y),i))   === (view(X,:,i), view(y,i))
#             @test @inferred(datasubset((Xv,y),i))  === (view(X,:,i), view(y,i))
#             @test @inferred(datasubset((X,yv),i))  === (view(X,:,i), view(y,i))
#             @test @inferred(datasubset((Xv,yv),i)) === (view(X,:,i), view(y,i))
#             @test @inferred(datasubset((XX,X,y),i)) === (view(XX,:,:,i), view(X,:,i),view(y,i))
#             # compare if obs match in tuple
#             x1, y1 = getobs(datasubset((X1,Y1), i))
#             @test all(x1' .== y1)
#             x1, y1, z1 = getobs(datasubset((X1,Y1,X1), i))
#             @test all(x1' .== y1)
#             @test all(x1 .== z1)
#         end
#     end

#     println("<HEARTBEAT>")

#     @testset "SparseArray" begin
#         @test @inferred(datasubset(Xs)) === DataSubset(Xs)
#         @test @inferred(datasubset(ys)) === DataSubset(ys)
#         for i in (1:150, 2:10, [2,5,7], [2,1])
#             @test @inferred(datasubset(Xs,i)) === DataSubset(Xs,i)
#             @test @inferred(datasubset(ys,i)) === DataSubset(ys,i)
#         end
#     end

#     @testset "Tuple of SparseArray" begin
#         @test @inferred(datasubset((Xv,ys))) === (Xv,DataSubset(ys))
#         @test @inferred(datasubset((X,ys)))  === (datasubset(X),DataSubset(ys))
#         @test @inferred(datasubset((Xs,y)))  === (DataSubset(Xs),datasubset(y))
#         @test @inferred(datasubset((Xs,ys))) === (DataSubset(Xs),DataSubset(ys))
#         @test @inferred(datasubset((Xs,Xs))) === (DataSubset(Xs),DataSubset(Xs))
#         @test @inferred(datasubset((ys,Xs))) === (DataSubset(ys),DataSubset(Xs))
#         @test @inferred(datasubset((XX,Xs,yv))) === (datasubset(XX),DataSubset(Xs),yv)
#         for i in (1:150, 2:10, [2,5,7], [2,1])
#             @test @inferred(datasubset((X,ys),i))  === (view(X,:,i), DataSubset(ys,i))
#             @test @inferred(datasubset((Xs,y),i))  === (DataSubset(Xs,i), view(y,i))
#             @test @inferred(datasubset((Xs,ys),i)) === (DataSubset(Xs,i), DataSubset(ys,i))
#             @test @inferred(datasubset((Xs,Xs),i)) === (DataSubset(Xs,i), DataSubset(Xs,i))
#             @test @inferred(datasubset((ys,Xs),i)) === (DataSubset(ys,i), DataSubset(Xs,i))
#             @test @inferred(datasubset((XX,Xs,y),i)) === (view(XX,:,:,i),DataSubset(Xs,i),view(y,i))
#             # compare if obs match in tuple
#             x1, y1 = getobs(datasubset((X1,sparse(Y1)), i))
#             @test all(x1' .== y1)
#             x1, y1, z1 = getobs(datasubset((X1,Y1,sparse(X1)), i))
#             @test all(x1' .== y1)
#             @test all(x1 .== z1)
#         end
#     end

#     @testset "custom types" begin
#         @test_throws MethodError datasubset(EmptyType())
#         @test_throws MethodError datasubset(EmptyType(), 1:10)
#         @test_throws MethodError datasubset(EmptyType(), 1:10, ObsDim.First())
#         @test_throws MethodError datasubset(CustomType(), obsdim=1)
#         @test_throws MethodError datasubset(CustomType(), obsdim=:last)
#         @test_throws MethodError datasubset(CustomType(), 2:10, obsdim=1)
#         @test_throws MethodError datasubset(CustomType(), 2:10, obsdim=:last)
#         @test_throws BoundsError getobs(datasubset(CustomType(), 11:20), 11)
#         @test typeof(@inferred(datasubset(CustomType()))) <: DataSubset
#         @test datasubset(CustomType()) == DataSubset(CustomType())
#         @test datasubset(CustomType(), 11:20) == DataSubset(CustomType(), 11:20)
#         @test nobs(datasubset(CustomType())) === 100
#         @test nobs(datasubset(CustomType(), 11:20)) === 10
#         @test getobs(datasubset(CustomType())) == collect(1:100)
#         @test getobs(datasubset(CustomType(), 11:20), 10) == 20
#         @test getobs(datasubset(CustomType(), 11:20), [3,5]) == [13,15]
#     end
# end

# @testset "getobs!" begin
#     @test getobs!(nothing, datasubset(y, 1)) == "setosa"

#     @testset "DataSubset" begin
#         xbuf1 = zeros(4,8)
#         s1 = DataSubset(X, 2:9)
#         @test @inferred(getobs!(xbuf1,s1)) === xbuf1
#         @test xbuf1 == getobs(s1)
#         xbuf1 = zeros(4,5)
#         s1 = DataSubset(X, 10:17)
#         @test @inferred(getobs!(xbuf1,s1,2:6)) === xbuf1
#         @test xbuf1 == getobs(s1,2:6) == getobs(X,11:15)

#         xbuf2 = zeros(8,4)
#         s2 = DataSubset(X', 2:9, obsdim=1)
#         @test @inferred(getobs!(xbuf2,s2)) === xbuf2
#         @test xbuf2 == getobs(s2)
#         xbuf2 = zeros(5,4)
#         s2 = DataSubset(X', 10:17, obsdim=1)
#         @test @inferred(getobs!(xbuf2,s2,2:6)) === xbuf2
#         @test xbuf2 == getobs(s2,2:6) == getobs(X',11:15,obsdim=1)

#         s3 = DataSubset(Xs, 11:15)
#         @test @inferred(getobs!(nothing,s3)) == getobs(Xs,11:15)

#         s4 = DataSubset(CustomType(), 6:10)
#         @test @inferred(getobs!(nothing,s4)) == getobs(s4)
#         s5 = DataSubset(CustomType(), 9:20)
#         @test @inferred(getobs!(nothing,s5,2:6)) == getobs(s5,2:6)
#     end

#     @testset "Tuple with DataSubset" begin
#         xbuf = zeros(4,2)
#         ybuf = ["foo", "bar"]
#         s1 = DataSubset(Xs, 5:9)
#         s2 = DataSubset(X, 5:9)
#         @test_throws AssertionError getobs!((nothing,xbuf),(s1,s2),2:3,ObsDim.First())
#         @test_throws AssertionError getobs!((nothing,xbuf),(s1,s2),2:3,(ObsDim.First(),ObsDim.Last()))
#         @test getobs!((nothing,xbuf),(s1,s2), 2:3) == (getobs(Xs,6:7),xbuf)
#         @test xbuf == getobs(X,6:7)
#         @test getobs!((nothing,xbuf),(s1,s2), 2:3, ObsDim.Last()) == (getobs(Xs,6:7),xbuf)
#         @test getobs!((nothing,xbuf),(s1,s2), 2:3, (ObsDim.Last(),ObsDim.Last())) == (getobs(Xs,6:7),xbuf)
#     end
# end
