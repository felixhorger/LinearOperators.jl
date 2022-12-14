
"""

This module lives off side effects.
It offers the possiblity for good performance though.

1) Side-effect free operator by omitting argument `y`, not recommended
```julia
A = LinearOperator{Float64}((10, 10), (y, x) -> I(10) * x)
```

2) Non-side-effect free operator, by providing an output vector, recommended
```julia
A = LinearOperator{Float64}((10, 10), (y, x) -> mul!(y, I(10), x); out=Vector{Float64}(undef, 10))
```

Keep in mind: if you want to reuse the result of an operator-vector product, copy it!

"""
module LinearOperators

	import Base.:*
	import Base.:+
	import Base.:-
	import Base.:^
	import Base.adjoint
	import Base.inv
	import Base.size
	import Base.show
	import LinearAlgebra.mul!
	import FunctionWrappers: FunctionWrapper

	export AbstractLinearOperator
	export LinearOperator, CompositeLinearOperator
	export HermitianOperator, UnitaryOperator, UniformScalingOperator

	# Function used as a placeholder
	dummy(x, y) = error("Not implemented by user")

	# Basic types of linear operators
	abstract type AbstractLinearOperator{T <: Number} end

	# Make it callable
	(A::AbstractLinearOperator{T})(x::AbstractVector{<: Number}) where T <: Number = A * x
	(A::AbstractLinearOperator{T})(y::AbstractVector{T}, x::AbstractVector{<: Number}) where T <: Number = A.op(y, x)

	# Function and pre-allocated output
	MatrixFunction{T} = FunctionWrapper{Vector{T}, NTuple{2, Vector{T}}}

	# n × m
	struct LinearOperator{T <: Number} <: AbstractLinearOperator{T}
		shape::NTuple{2, Integer}
		op::MatrixFunction{T}
		adj::MatrixFunction{T}
		inv::MatrixFunction{T}
		invadj::MatrixFunction{T}
		out::Vector{T}
		out_adj_inv::Vector{T}
		function LinearOperator{T}(
			shape, op;
			adj=dummy, inv=dummy, invadj=dummy,
			out=Vector{T}(undef, 0),
			out_adj_inv=Vector{T}(undef, 0)
		) where T
			new{T}(shape, op, adj, inv, invadj, out, out_adj_inv)
		end
	end
	struct HermitianOperator{T <: Number} <: AbstractLinearOperator{T}
		dim::Int
		op::MatrixFunction{T}
		inv::MatrixFunction{T}
		out::Vector{T}
		HermitianOperator{T}(dim, op; inv=dummy, out=Vector{T}(undef, 0)) where T = new{T}(dim, op, inv, out)
	end
	struct UnitaryOperator{T <: Number} <: AbstractLinearOperator{T}
		dim::Integer
		op::MatrixFunction{T}
		adj::MatrixFunction{T}
		out::Vector{T}
		UnitaryOperator{T}(dim, op; adj=dummy, out=Vector{T}(undef, 0)) where T = new{T}(dim, op, adj, out)
	end
	struct UniformScalingOperator{T <: Number} <: AbstractLinearOperator{T}
		dim::Integer
		scalar::T
		out::Vector{T}
		UniformScalingOperator{T}(dim, scalar::T; out=Vector{T}(undef, 0)) where T = new{T}(dim, scalar, out)
	end

	# Composing linear operators
	struct CompositeLinearOperator{T, N, O} <: AbstractLinearOperator{T}
		ops::NTuple{N, AbstractLinearOperator{T}}
	end

	# Size
	issquare(A::AbstractLinearOperator) = true
	issquare(A::LinearOperator) = (A.shape[1] == A.shape[2])
	issquare(A::CompositeLinearOperator{T, N, :+}) where {T, N} = (A[1].shape[1] == A[1].shape[2])
	issquare(A::CompositeLinearOperator{T, N, :*}) where {T, N} = (size(A.ops[1], 1) == size(A.ops[N], 2))
	function size(A::AbstractLinearOperator, d::Integer)
		(1 > d > 2) && throw(BoundsError("d must be in (1,2)"))
		return A.dim
	end
	size(A::AbstractLinearOperator)		= (A.dim, A.dim)
	size(A::LinearOperator)				= A.shape
	size(A::LinearOperator, d::Integer)	= A.shape[d]
	size(A::CompositeLinearOperator{T, N, :*})				where {T, N} = (size(A.ops[1], 1), size(A.ops[N], 2))
	size(A::CompositeLinearOperator{T, N, :*}, d::Integer)	where {T, N} = size(A)[d]
	size(A::CompositeLinearOperator{T, N, :+})				where {T, N} = size(A.ops[1])
	size(A::CompositeLinearOperator{T, N, :+}, d::Integer)	where {T, N} = size(A.ops[1])[d]

	# Multiplication and summation
	function checkdims(op::Val{:*}, A::AbstractLinearOperator{T}, B::AbstractLinearOperator{T}) where T
		nA = size(A, 2)
		nB = size(B, 1)
		nA != nB && throw(DimensionMismatch("second dimension of A, $nA, does not match first dimension of B, $nB"))
		return
	end
	function checkdims(op::Val{:+}, A::AbstractLinearOperator{T}, B::AbstractLinearOperator{T}) where T
		nA = size(A)
		nB = size(B)
		nA != nB && throw(DimensionMismatch("dimensions of A, $nA, do not match dimensions of B, $nB"))
		return
	end
	for ⨀ in (:*, :+)
		sym_⨀ = Expr(:quote, ⨀)
		@eval begin
			# A ⨀ B
			function $⨀(A::AbstractLinearOperator{T}, B::AbstractLinearOperator{T}) where T
				checkdims(Val($sym_⨀), A, B)
				return CompositeLinearOperator{T, 2, $sym_⨀}((A, B))
			end
			# (A₁ ⨀ A₂ ...) ⨀ B
			function $⨀(A::CompositeLinearOperator{T, N, $sym_⨀}, B::AbstractLinearOperator{T}) where {T, N}
				checkdims(Val($sym_⨀), A, B)
				return CompositeLinearOperator{T, N+1, $sym_⨀}((A.ops..., B))
			end
			# B ⨀ (A₁ ⨀ A₂ ...)
			function $⨀(A::AbstractLinearOperator{T}, B::CompositeLinearOperator{T, N, $sym_⨀}) where {T, N}
				checkdims(Val($sym_⨀), A, B)
				return CompositeLinearOperator{T, N+1, $sym_⨀}((A, B.ops...))
			end
			# (A₁ ⨀ A₂ ...) ⨀ (B₁ ⨀ B₂ ...)
			function $⨀(A::CompositeLinearOperator{T, N, $sym_⨀}, B::CompositeLinearOperator{T, M, $sym_⨀}) where {T, N, M}
				checkdims(Val($sym_⨀), A, B)
				return CompositeLinearOperator{T, N+M, $sym_⨀}((A.ops..., B.ops...))
			end
		end
	end
	-(A::AbstractLinearOperator{T}, B::AbstractLinearOperator{T}) where T = A + (-B)
	-(A::AbstractLinearOperator{T}) where T = (-one(T)) * A
	# Mixing summation and multiplication
	for (○, ⨀) in ((:*, :+), (:+, :*))
		sym_⨀ = Expr(:quote, ⨀)
		sym_○ = Expr(:quote, ○)
		@eval begin
			# (A₁ ⨀ A₂ ...) ○ (B₁ ⨀ B₂ ...)
			function $○(A::CompositeLinearOperator{T, N, $sym_⨀}, B::CompositeLinearOperator{T, M, $sym_⨀}) where {T, N, M}
				checkdims(Val($sym_○), A, B)
				return CompositeLinearOperator{T, 2, $sym_○}((A, B))
			end
			# (A₁ ○ A₂ ...) ○ (B₁ ⨀ B₂ ...)
			function $○(A::CompositeLinearOperator{T, N, $sym_○}, B::CompositeLinearOperator{T, M, $sym_⨀}) where {T, N, M}
				checkdims(Val($sym_○), A, B)
				return CompositeLinearOperator{T, N+1, $sym_○}((A.ops..., B))
			end
			# (A₁ ⨀ A₂ ...) ○ (B₁ ○ B₂ ...)
			function $○(A::CompositeLinearOperator{T, N, $sym_⨀}, B::CompositeLinearOperator{T, M, $sym_○}) where {T, N, M}
				checkdims(Val($sym_○), A, B)
				return CompositeLinearOperator{T, N+1, $sym_○}((A, B.ops...))
			end
		end
	end

	# Operations on linear operators
	# Adjoint
	adjoint(A::LinearOperator{T}) where T = LinearOperator{T}(
		reverse(A.shape),
		A.adj;
		adj		= A.op,
		inv		= A.invadj,
		invadj	= A.inv,
		out		= A.out_adj_inv,
		out_adj_inv = A.out
	)
	adjoint(A::HermitianOperator)			= A
	adjoint(A::UnitaryOperator{T}) where T	= UnitaryOperator{T}(A.dim, A.adj; adj=A.op, out=A.out)
	adjoint(A::UniformScalingOperator)		= A
	# Inverse
	function inv(A::LinearOperator{T}) where T
		@assert issquare(A)
		return LinearOperator{T}(
			A.shape, A.inv;
			adj=A.invadj,
			inv=A.op,
			invadj=A.adj,
			out=A.out_adj_inv,
			out_inv_adj=A.out
		)
	end
	inv(A::HermitianOperator{T})		where T = HermitianOperator{T}(A.dim, A.inv; inv=A.op, out=A.out)
	inv(A::UnitaryOperator{T})			where T = UnitaryOperator{T}(A.dim, A.adj; adj=A.op, out=A.out)
	inv(A::UniformScalingOperator{T})	where T = UniformScalingOperator{T}(A.dim, 1 ./ A.scalar; out=A.out)
	# Both for composite type
	# Note: inv will only work if all operators are square
	for op in (:adjoint, :inv)
		@eval begin
			$op(A::CompositeLinearOperator{T, N, :*}) where {T, N} = CompositeLinearOperator{T, N, :*}($op.(reverse(A.ops)))
			$op(A::CompositeLinearOperator{T, N, :+}) where {T, N} = CompositeLinearOperator{T, N, :+}($op.(A.ops))
		end
	end

	# Potentiation
	function ^(A::AbstractLinearOperator{T}, p::Integer) where T
		@assert p > 0
		@assert issquare(A)
		return CompositeLinearOperator{T, p, :*}(Tuple(A for i = 1:p))
		# Note: output type depends on runtime argument
	end

	# Matrix vector multiplication
	function checkdims(op::Val{:*}, A::AbstractLinearOperator, x::AbstractVector)
		nA = size(A, 2)
		nx = length(x)
		nA != nx && throw(DimensionMismatch("second dimension of A, $nA, does not match length of x, $nx"))
		return
	end
	function *(A::AbstractLinearOperator{T}, x::AbstractVector{T}) where T <: Number
		checkdims(Val(:*), A, x)
		return A.op(A.out, x)
	end
	function *(A::CompositeLinearOperator{T, N, :*}, x::AbstractVector{T}) where {T <: Number, N}
		checkdims(Val(:*), A, x)
		y = x
		for B in reverse(A.ops)
			y = B * y # y stays a Vector here, i.e. type-stable
		end
		return y
	end
	function *(A::CompositeLinearOperator{T, N, :+}, x::AbstractVector{T}) where {T <: Number, N}
		checkdims(Val(:*), A, x)
		y = A.ops[1] * x
		for B in A.ops[2:end]
			y .+= B * x
		end
		return y
	end
	mul!(y::AbstractVector{T}, A::AbstractLinearOperator{T}, x::AbstractVector{T}) where T <: Number = A.op(y, x)

	# If input types are wrong
	*(A::AbstractLinearOperator{T}, x::AbstractVector{<: Number}) where T = A * convert.(T, x)
	mul!(y::AbstractVector{T}, A::AbstractLinearOperator{T}, x::AbstractVector{<: Number}) where T = mul!(y, A, convert.(T, x))

	# Matrix scalar multiplication
	*(A::AbstractLinearOperator{T}, a::T) where T = CompositeLinearOperator{T, 2, :*}((A, UniformScalingOperator{T}(size(A, 2), a)))
	*(a::T, A::AbstractLinearOperator{T}) where T = CompositeLinearOperator{T, 2, :*}((UniformScalingOperator{T}(size(A, 1), a), A))
	*(a::Number, A::AbstractLinearOperator{T}) where T = convert(T, a) * A
	*(A::AbstractLinearOperator{T}, a::Number) where T = A * convert(T, a)
	# Note: Despite the equivalence A * a == a * A, if A is non-square it can have a performance impact

	# Distinguishing types
	for f in (:all_hermitian_type, :all_unitary_type)
		@eval begin
			function $f(A::CompositeLinearOperator)
				is_it = true
				for B in A.ops
					$f(A) && continue
					is_it = false
					break
				end
				return is_it
			end
		end
	end

	show(io::IO, A::AbstractLinearOperator) = print(io, "$(typeof(A))(...)")
	show(io::IO, ::MIME"text/plain", A::AbstractLinearOperator) = show(io, A)
end

