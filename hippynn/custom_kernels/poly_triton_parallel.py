"""Polynomial-parallel Triton evaluator.

Unlike the original backend, the second launch dimension assigns one Triton
program to each polynomial, allowing different polynomials to run in parallel.
"""

import triton
import triton.language as tl


def evaluate_polynomials_parallel(
    x,
    coefs,
    terms,
    polynomial_sizes,
    polynomial_offsets,
    output,
    num_polynomials,
    num_monomials,
    input_dimension_rounded_up,
    num_points,
    degree_rounded_up,
    point_bucket,
    dtype,
):
    """Launch the two-dimensional polynomial-parallel kernel."""
    grid = lambda meta: (
        triton.cdiv(num_points, meta["NUM_POINTS_TO_LOAD"]),
        num_polynomials,
    )
    evaluate_polynomials_parallel_kernel[grid](
        x,
        coefs,
        terms,
        polynomial_sizes,
        polynomial_offsets,
        output,
        num_polynomials,
        num_monomials,
        input_dimension_rounded_up,
        num_points,
        degree_rounded_up,
        point_bucket=point_bucket,
        dtype=dtype,
    )


def get_configs_polynomials():
    """Return the configurations searched when autotuning the kernel."""
    # (NUM_POINTS_TO_LOAD, NUM_MONOMIALS_TO_LOAD, num_warps, num_stages)
    # Keep num_warps * 32 <= NUM_POINTS_TO_LOAD; configs with more threads than
    # points fail to compile for small bases.
    candidates = (
        (32, 16, 1, 2),
        (32, 32, 1, 2),
        (64, 16, 2, 2),
        (64, 32, 2, 2),
    )
    return [
        triton.Config(
            kwargs={
                "NUM_POINTS_TO_LOAD": num_points,
                "NUM_MONOMIALS_TO_LOAD": num_monomials,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for num_points, num_monomials, num_warps, num_stages in candidates
    ]


@triton.jit
def prod(x, y):
    return x * y


@triton.autotune(
    configs=get_configs_polynomials(),
    key=["point_bucket", "num_monomials", "num_polynomials"],
)
@triton.jit
def evaluate_polynomials_parallel_kernel(
    input_ptr,
    coefs_ptr,
    terms_ptr,
    polynomial_sizes_ptr,
    polynomial_offsets_ptr,
    output_ptr,
    num_polynomials: tl.constexpr,
    num_monomials: tl.constexpr,
    input_dimension_rounded_up: tl.constexpr,
    num_points,
    degree_rounded_up: tl.constexpr,
    point_bucket: tl.constexpr,
    NUM_POINTS_TO_LOAD: tl.constexpr,
    NUM_MONOMIALS_TO_LOAD: tl.constexpr,
    dtype: tl.constexpr = tl.float32,
):
    """
    Triton kernel that is used to simultaneously evaluate many multivariate polynomials on a batch of points.
    Each program evaluates a single polynomial on a block of points, so that different polynomials are evaluated in parallel.

    :param input_ptr: Pointer to the array of input points.

    :param coefs_ptr: Pointer to the array of coefficients of the monomials.

    :param terms_ptr: Pointer to the array of terms. The number of columns MUST be a power of 2. Pad using -1 if necessary.

    :param polynomial_sizes_ptr: Pointer to the array of polynomial sizes.

    :param polynomial_offsets_ptr: Pointer to the array of polynomial offsets. Entry j is the index of the first monomial of polynomial p_j,
                       i.e. the exclusive cumulative sum of the polynomial sizes.

    :param output_ptr: Pointer to the array that will store the output values. Each row corresponds to a different input point X_i. Each column corresponds to a different polynomial p_j.
                       Thus, the ij entry of the output_ptr will be p_j(X_i).

    :param num_monomials: The total number of monomials across all polynomials. This should equal len(coefs_ptr).

    :param input_dimension_rounded_up: Let n represent the number of entries of each X_i (so for (1,2,3), n=3). Then input_dimension_rounded_up is the smallest power of 2 that is larger than n.

    :param num_points: The number of points X_i, equal to the number of rows in the tensor pointed to by input_ptr.

    :param degree_rounded_up: Let n represent the maximum degree of all of the polynomials. Then degree_rounded_up is the smallest power of 2 that is larger than n. This is also the number of 
                        columns in the tensor pointed to by terms_ptr.

    :param point_bucket: num_points rounded up to a power of 2. This is only used as an autotuning key, so that the kernel is tuned separately for different batch sizes.

    :param NUM_POINTS_TO_LOAD: hyperparameter. This specifies the number of points to evaluate monomials on at the same time.

    :param NUM_MONOMIALS_TO_LOAD: hyperparameter. This specifies the number of monomials to evaluate at the same time.

    :param dtype: Specifies whether the output will be float32 or float64.
    """
    x_pid = tl.program_id(0)
    poly = tl.program_id(1)

    x_start = x_pid * NUM_POINTS_TO_LOAD * input_dimension_rounded_up
    x_offsets = x_start + tl.arange(
        0, NUM_POINTS_TO_LOAD * input_dimension_rounded_up
    )
    x_mask = x_offsets < num_points * input_dimension_rounded_up
    x = tl.load(input_ptr + x_offsets, mask=x_mask)
    x = x.reshape((NUM_POINTS_TO_LOAD, input_dimension_rounded_up))

    point_offsets = x_pid * NUM_POINTS_TO_LOAD + tl.arange(0, NUM_POINTS_TO_LOAD)
    point_mask = point_offsets < num_points

    polynomial_size = tl.load(polynomial_sizes_ptr + poly)
    polynomial_start = tl.load(polynomial_offsets_ptr + poly)
    output = tl.zeros((NUM_POINTS_TO_LOAD,), dtype=dtype)
    monomial_lanes = tl.arange(0, NUM_MONOMIALS_TO_LOAD)
    monomial_indexer = monomial_lanes[None, :, None]
    num_monomial_loops = tl.cdiv(polynomial_size, NUM_MONOMIALS_TO_LOAD)

    for monomial_idx in range(num_monomial_loops):
        local_monomials = (
            monomial_idx * NUM_MONOMIALS_TO_LOAD + monomial_lanes
        )
        monomial_mask = local_monomials < polynomial_size
        monomial_offsets = polynomial_start + local_monomials

        coef = tl.load(
            coefs_ptr + monomial_offsets, mask=monomial_mask
        )[None, :]
        coef = coef.broadcast_to(
            (NUM_POINTS_TO_LOAD, NUM_MONOMIALS_TO_LOAD)
        )

        term_lanes = tl.arange(
            0, NUM_MONOMIALS_TO_LOAD * degree_rounded_up
        )
        terms_offsets = polynomial_start * degree_rounded_up + term_lanes
        terms_offsets += (
            monomial_idx * NUM_MONOMIALS_TO_LOAD * degree_rounded_up
        )
        terms_mask = term_lanes < (
            polynomial_size - monomial_idx * NUM_MONOMIALS_TO_LOAD
        ) * degree_rounded_up
        terms_index = tl.load(
            terms_ptr + terms_offsets, mask=terms_mask, other=-1
        )
        terms_index = terms_index.reshape(
            (1, NUM_MONOMIALS_TO_LOAD * degree_rounded_up)
        )
        terms_index = terms_index.broadcast_to(
            (
                NUM_POINTS_TO_LOAD,
                NUM_MONOMIALS_TO_LOAD * degree_rounded_up,
            )
        )

        terms = tl.gather(x, terms_index, 1)
        terms = tl.where(terms_index >= 0, terms, 1)
        terms = terms.reshape(
            (
                NUM_POINTS_TO_LOAD,
                NUM_MONOMIALS_TO_LOAD,
                degree_rounded_up,
            )
        )
        terms = tl.where(
            monomial_idx * NUM_MONOMIALS_TO_LOAD + monomial_indexer
            < polynomial_size,
            terms,
            0,
        )
        terms = tl.reduce(terms, 2, prod)
        output += tl.sum(terms * coef, axis=1)

    output_offsets = point_offsets * num_polynomials + poly
    tl.store(output_ptr + output_offsets, output, mask=point_mask)
