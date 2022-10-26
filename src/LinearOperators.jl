
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

	export AbstractLinearOperator
	export LinearOperator, CompositeLinearOperator
	export HermitianOperator, UnitaryOperator, UniformScalingOperator

	# Function used as a placeholder
	dummy(x) = error("Not implemented by user") 

	# Basic types of linear operators
	abstract type AbstractLinearOperator end

	# Make it callable
	(A::AbstractLinearOperator)(x) = A * x

	# n × m
	struct LinearOperator <: AbstractLinearOperator
		shape::NTuple{2, Integer}
		op::Function
		adj::Function
		inv::Function
		invadj::Function
		LinearOperator(shape::NTuple{2, Integer}, op, adj=dummy, inv=dummy, invadj=dummy) = new(shape, op, adj, inv, invadj)
	end
	struct HermitianOperator <: AbstractLinearOperator
		dim::Integer
		op::Function
		inv::Function
		HermitianOperator(dim, op, inv=dummy) = new(dim, op, inv)
	end
	struct UnitaryOperator <: AbstractLinearOperator
		dim::Integer
		op::Function
		adj::Function
		UnitaryOperator(dim, op, adj=dummy) = new(dim, op, adj)
	end
	struct UniformScalingOperator <: AbstractLinearOperator
		dim::Integer
		scalar::Number
	end

	# Composing linear operators
	struct CompositeLinearOperator{N,T} <: AbstractLinearOperator where {N,T}
		ops::NTuple{N, AbstractLinearOperator}
	end

	# Size
	issquare(A::AbstractLinearOperator) = true
	issquare(A::LinearOperator) = (A.shape[1] == A.shape[2])
	issquare(A::CompositeLinearOperator) = (A[1].shape[1] == A[1].shape[2])
	issquare(A::CompositeLinearOperator{N, :*}) where {N,T} = (size(A.ops[1], 1) == size(A.ops[N], 2))
	function size(A::AbstractLinearOperator, d::Integer)
		(1 > d > 2) && throw(BoundsError("d must be in (1,2)"))
		return A.dim
	end
	size(A::AbstractLinearOperator) = (A.dim, A.dim)
	size(A::LinearOperator) = A.shape
	size(A::LinearOperator, d::Integer) = A.shape[d]
	size(A::CompositeLinearOperator{N, :*}) where N = (size(A.ops[1], 1), size(A.ops[N], 2))
	size(A::CompositeLinearOperator{N, :*}, d::Integer) where N = size(A)[d]
	size(A::CompositeLinearOperator{N, :+})  where N = size(A.ops[1])
	size(A::CompositeLinearOperator{N, :+}, d::Integer) where N = size(A.ops[1])[d]

	# Multiplication and summation
	function checkdims(op::Val{:*}, A::AbstractLinearOperator, B::AbstractLinearOperator)
		nA = size(A, 2)
		nB = size(B, 1)
		nA != nB && throw(DimensionMismatch("second dimension of A, $nA, does not match first dimension of B, $nB"))
		return
	end
	function checkdims(op::Val{:+}, A::AbstractLinearOperator, B::AbstractLinearOperator)
		nA = size(A)
		nB = size(B)
		nA != nB && throw(DimensionMismatch("dimensions of A, $nA, do not match dimensions of B, $nB"))
		return
	end
	for ⨀ in (:*, :+)
		sym_⨀ = Expr(:quote, ⨀)
		@eval begin
			# A ⨀ B
			function $⨀(A::AbstractLinearOperator, B::AbstractLinearOperator)
				checkdims(Val($sym_⨀), A, B)
				return CompositeLinearOperator{2, $sym_⨀}((A, B))
			end
			# (A₁ ⨀ A₂ ...) ⨀ B
			function $⨀(A::CompositeLinearOperator{N, $sym_⨀}, B::AbstractLinearOperator) where N
				checkdims(Val($sym_⨀), A, B)
				return CompositeLinearOperator{N+1, $sym_⨀}((A.ops..., B))
			end
			# B ⨀ (A₁ ⨀ A₂ ...)
			function $⨀(A::AbstractLinearOperator, B::CompositeLinearOperator{N, $sym_⨀}) where N
				checkdims(Val($sym_⨀), A, B)
				return CompositeLinearOperator{N+1, $sym_⨀}((A, B.ops...))
			end
			# (A₁ ⨀ A₂ ...) ⨀ (B₁ ⨀ B₂ ...)
			function $⨀(A::CompositeLinearOperator{N, $sym_⨀}, B::CompositeLinearOperator{M, $sym_⨀}) where {N,M}
				checkdims(Val($sym_⨀), A, B)
				return CompositeLinearOperator{N+M, $sym_⨀}((A.ops..., B.ops...))
			end
		end
	end
	-(A::AbstractLinearOperator, B::AbstractLinearOperator) = A + (-B)
	-(A::AbstractLinearOperator) = (-1) * A
	# Mixing summation and multiplication
	for (○, ⨀) in ((:*, :+), (:+, :*))
		sym_⨀ = Expr(:quote, ⨀)
		sym_○ = Expr(:quote, ○)
		@eval begin
			# (A₁ ⨀ A₂ ...) ○ (B₁ ⨀ B₂ ...)
			function $○(A::CompositeLinearOperator{N, $sym_⨀}, B::CompositeLinearOperator{M, $sym_⨀}) where {N,M}
				checkdims(Val($sym_○), A, B)
				return CompositeLinearOperator{2, $sym_○}((A, B))
			end
			# (A₁ ○ A₂ ...) ○ (B₁ ⨀ B₂ ...)
			function $○(A::CompositeLinearOperator{N, $sym_○}, B::CompositeLinearOperator{M, $sym_⨀}) where {N,M}
				checkdims(Val($sym_○), A, B)
				return CompositeLinearOperator{N+1, $sym_○}((A.ops..., B))
			end
			# (A₁ ⨀ A₂ ...) ○ (B₁ ○ B₂ ...)
			function $○(A::CompositeLinearOperator{N, $sym_⨀}, B::CompositeLinearOperator{M, $sym_○}) where {N,M}
				checkdims(Val($sym_○), A, B)
				return CompositeLinearOperator{N+1, $sym_○}((A, B.ops...))
			end
		end
	end

	# Operations on linear operators
	# Adjoint
	adjoint(A::LinearOperator) = LinearOperator(reverse(A.shape), A.adj, A.op, A.invadj, A.inv)
	adjoint(A::HermitianOperator) = A
	adjoint(A::UnitaryOperator) = UnitaryOperator(A.dim, A.adj, A.op)
	adjoint(A::UniformScalingOperator) = A
	# Inverse
	function inv(A::LinearOperator)
		@assert issquare(A)
		return LinearOperator(A.shape, A.inv, A.invadj, A.op, A.adj)
	end
	inv(A::HermitianOperator) = HermitianOperator(A.dim, A.inv, A.op)
	inv(A::UnitaryOperator) = UnitaryOperator(A.dim, A.adj, A.op)
	inv(A::UniformScalingOperator) = UniformScalingOperator(A.dim, 1 ./ A.scalar)
	# Both for composite type
	# Note: inv will only work if all operators are square
	for op in (:adjoint, :inv)
		@eval begin
			$op(A::CompositeLinearOperator{N, :*}) where N = CompositeLinearOperator{N, :*}($op.(reverse(A.ops)))
			$op(A::CompositeLinearOperator{N, :+}) where N = CompositeLinearOperator{N, :+}($op.(A.ops))
		end
	end

	# Potentiation
	function ^(A::AbstractLinearOperator, p::Integer)
		@assert p > 0
		@assert issquare(A)
		return CompositeLinearOperator{p, :*}(Tuple(A for i = 1:p))
		# Note: output type depends on runtime argument
	end

	# Matrix vector multiplication
	function checkdims(op::Val{:*}, A::AbstractLinearOperator, x::AbstractArray)
		nA = size(A, 2)
		nx = length(x)
		nA != nx && throw(DimensionMismatch("second dimension of A, $nA, does not match length of x, $nx"))
		return
	end
	function *(A::AbstractLinearOperator, x::AbstractArray)
		checkdims(Val(:*), A, x)
		return A.op(x)
	end
	function *(A::CompositeLinearOperator{N, :*}, x::AbstractArray) where N
		checkdims(Val(:*), A, x)
		y = x
		for B in reverse(A.ops)
			y = B * y
		end
		return y
	end
	function *(A::CompositeLinearOperator{N, :+}, x::AbstractArray) where N
		checkdims(Val(:*), A, x)
		y = zeros(eltype(x), size(x))
		for B in A.ops
			y .+= B * x
		end
		return y
	end
	function mul!(y, A::AbstractLinearOperator, x)
		z = A * x
		if y !== z # If they are not the same reference
			y .= z
		end
		return y
	end

	# Matrix scalar multiplication
	*(A::AbstractLinearOperator, a::Number) = CompositeLinearOperator{2, :*}((A, UniformScalingOperator(size(A, 2), a)))
	*(a::Number, A::AbstractLinearOperator) = CompositeLinearOperator{2, :*}((UniformScalingOperator(size(A, 1), a), A))

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

