using Test
using StatsBase
using LearnBase
using MLLabelUtils
using MLDataPattern
using ReferenceTests
using Random
using SparseArrays
using DataFrames

# --------------------------------------------------------------------
# create some test data

showcompact(io, x) = show(IOContext(io, :compact => true), x)

Random.seed!(1335)
X = rand(4, 150)
y = repeat(["setosa","versicolor","virginica"], inner = 50)
Y = permutedims(hcat(y,y), [2,1])
Yt = hcat(y,y)
yt = Y[1:1,:]
Xv = view(X,:,:)
yv = view(y,:)
XX = rand(20,30,150)
XXX = rand(3,20,30,150)
vars = (X, Xv, yv, XX, XXX, y)
tuples = ((X,y), (X,Y), (XX,X,y), (XXX,XX,X,y))
Xs = sprand(10,150,.5)
ys = sprand(150,.5)
# to compare if obs match
X1 = hcat((1:150 for i = 1:10)...)'
Y1 = collect(1:150)

struct EmptyType end

struct CustomType end
StatsBase.nobs(x::CustomType; obsdim = LearnBase.default_obsdim(x)) = 100
LearnBase.getobs(x::CustomType, i::Int; obsdim = LearnBase.default_obsdim(x)) = i
LearnBase.getobs(x::CustomType, i::AbstractVector; obsdim = LearnBase.default_obsdim(x)) = collect(i)
LearnBase.gettargets(::CustomType, i::Int) = "obs $i"
LearnBase.gettargets(::CustomType, i::AbstractVector) = "batch $i"

struct CustomStorage end
struct CustomObs{T}; data::T end
StatsBase.nobs(x::CustomStorage; obsdim = LearnBase.default_obsdim(x)) = 2
LearnBase.getobs(x::CustomStorage, i; obsdim = LearnBase.default_obsdim(x)) = CustomObs(i)
LearnBase.gettarget(str::String, obs::CustomObs) = "$str - obs $(obs.data)"
LearnBase.gettarget(obs::CustomObs) = "obs $(obs.data)"

struct ObsDimTriggeredException <: Exception end
struct MetaDataStorage end
StatsBase.nobs(x::MetaDataStorage; obsdim = LearnBase.default_obsdim(x)) = 3
LearnBase.getobs(x::MetaDataStorage, i; obsdim = LearnBase.default_obsdim(x)) = throw(ObsDimTriggeredException())
LearnBase.gettargets(::MetaDataStorage) = "full"
LearnBase.gettargets(::MetaDataStorage, i::Int) = "obs $i"
LearnBase.gettargets(::MetaDataStorage, i::AbstractVector) = "batch $i"

# --------------------------------------------------------------------

function matrix_compat_isequal(ref, actual)
    # a over-verbose collection of patterns that we want to ignore during test
    patterns = [
        # Julia v1.6
        "Normed{UInt8,8}" => "N0f8",
        r"Array{(\w+),2}" => s"Matrix{\1}",
        r"Array{(\w+),1}" => s"Vector{\1}",

        # https://github.com/JuliaGraphics/ColorTypes.jl/pull/206
        # r"Gray{\w+}\(([\w\.]+)\)" => s"\1",
        # r"RGB{\w+}\(([\w\.,]+)\)" => s"RGB(\1)",
    ]

    for p in patterns
        actual = replace(actual, p)
        ref = replace(ref, p)
    end

    # Julia v1.4
    ref = join(map(strip, split(ref, "\n")), "\n")
    actual = join(map(strip, split(actual, "\n")), "\n")

    isequal(ref, actual)
end

tests = [
    "tst_container.jl"
    "tst_datasubset.jl"
    "tst_randobs.jl"
    "tst_shuffleobs.jl"
    # "tst_splitobs.jl"
    # "tst_dataview.jl"
    # "tst_slidingwindow.jl"
    # "tst_targets.jl"
    # "tst_stratifiedobs.jl"
    # "tst_resample.jl"
    # "tst_folds.jl"
    # "tst_dataiterator.jl"
    "tst_dataframes.jl"
]

for t in tests
    @testset "$t" begin
        include(t)
    end
end
