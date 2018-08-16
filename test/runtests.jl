using Test
using StatsBase
using LearnBase
using MLLabelUtils
using MLDataPattern
using ReferenceTests

# --------------------------------------------------------------------
# create some test data

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
LearnBase.nobs(::CustomType) = 100
LearnBase.getobs(::CustomType, i::Int) = i
LearnBase.getobs(::CustomType, i::AbstractVector) = collect(i)
LearnBase.gettargets(::CustomType, i::Int) = "obs $i"
LearnBase.gettargets(::CustomType, i::AbstractVector) = "batch $i"

struct CustomStorage end
struct CustomObs{T}; data::T end
LearnBase.nobs(::CustomStorage) = 2
LearnBase.getobs(::CustomStorage, i) = CustomObs(i)
LearnBase.gettarget(str::String, obs::CustomObs) = "$str - obs $(obs.data)"
LearnBase.gettarget(obs::CustomObs) = "obs $(obs.data)"

struct ObsDimTriggeredException <: Exception end
struct MetaDataStorage end
LearnBase.nobs(::MetaDataStorage) = 3
LearnBase.getobs(::MetaDataStorage, i) = throw(ObsDimTriggeredException())
LearnBase.gettargets(::MetaDataStorage) = "full"
LearnBase.gettargets(::MetaDataStorage, i::Int) = "obs $i"
LearnBase.gettargets(::MetaDataStorage, i::AbstractVector) = "batch $i"

# --------------------------------------------------------------------

tests = [
    "tst_container.jl"
    "tst_datasubset.jl"
    "tst_randobs.jl"
    "tst_shuffleobs.jl"
    "tst_splitobs.jl"
    "tst_dataview.jl"
    "tst_slidingwindow.jl"
    "tst_targets.jl"
    "tst_stratifiedobs.jl"
    "tst_resample.jl"
    "tst_folds.jl"
    "tst_dataiterator.jl"
]

for t in tests
    @testset "$t" begin
        include(t)
    end
end
