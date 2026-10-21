"""Original Triton polynomial evaluator.

This backend intentionally preserves the one-dimensional launch and sequential
loop over polynomials used before the polynomial-parallel optimization.
"""

import triton
import triton.language as tl


def evaluate_polynomials_sequential(
    x,
    coefs,
    terms,
    polynomial_sizes,
    output,
    num_polynomials,
    num_monomials,
    input_dimension_rounded_up,
    num_points,
    degree_rounded_up,
    dtype,
):
    """Launch the original sequential-over-polynomials kernel."""
    grid = lambda meta: (
        triton.cdiv(num_points, meta["NUM_POINTS_TO_LOAD"]),
    )
    evaluate_polynomials_sequential_kernel[grid](
        x,
        coefs,
        terms,
        polynomial_sizes,
        output,
        num_polynomials,
        num_monomials,
        input_dimension_rounded_up,
        num_points,
        degree_rounded_up,
        dtype=dtype,
    )


def get_configs_polynomials():
    """Return the configurations searched when autotuning the kernel."""
    # I have arbitrarily chosen some safe, general autotune configs here. In the
    # future you could tweak this to be better for more niche applications.
    # (NUM_POINTS_TO_LOAD, NUM_MONOMIALS_TO_LOAD, num_warps, num_stages)
    # Keep num_warps * 32 <= NUM_POINTS_TO_LOAD; configs with more threads than
    # points fail to compile for small bases.
    candidates = (
        (128, 8, 2, 2),
        (128, 16, 2, 2),
        (128, 16, 4, 2),
        (128, 32, 4, 2),
        (256, 8, 4, 2),
        (256, 16, 4, 2),
        (256, 16, 8, 2),
        (256, 32, 8, 2),
        (512, 16, 8, 2),
        (512, 32, 16, 2),
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


@triton.autotune(configs=get_configs_polynomials(), key=["num_monomials"])
@triton.jit
def evaluate_polynomials_sequential_kernel(
    input_ptr,
    coefs_ptr,
    terms_ptr,
    polynomial_sizes_ptr,
    output_ptr,
    num_polynomials: tl.constexpr,
    num_monomials: tl.constexpr,
    input_dimension_rounded_up: tl.constexpr,
    num_points,
    degree_rounded_up: tl.constexpr,
    NUM_POINTS_TO_LOAD: tl.constexpr,
    NUM_MONOMIALS_TO_LOAD: tl.constexpr,
    dtype: tl.constexpr = tl.float32,
):
    """
    Triton kernel that is used to simultaneously evaluate many multivariate polynomials on a batch of points.
    Each program handles a block of points and loops over the polynomials sequentially.

    :param input_ptr: Pointer to the array of input points.

    :param coefs_ptr: Pointer to the array of coefficients of the monomials.

    :param terms_ptr: Pointer to the array of terms. The number of columns MUST be a power of 2. Pad using -1 if necessary.

    :param polynomial_sizes_ptr: Pointer to the array of polynomial sizes.

    :param output_ptr: Pointer to the array that will store the output values. Each row corresponds to a different input point X_i. Each column corresponds to a different polynomial p_j.
                       Thus, the ij entry of the output_ptr will be p_j(X_i).

    :param num_monomials: The total number of monomials across all polynomials. This should equal len(coefs_ptr).

    :param input_dimension_rounded_up: Let n represent the number of entries of each X_i (so for (1,2,3), n=3). Then input_dimension_rounded_up is the smallest power of 2 that is larger than n.

    :param num_points: The number of points X_i, equal to the number of rows in the tensor pointed to by input_ptr.

    :param degree_rounded_up: Let n represent the maximum degree of all of the polynomials. Then degree_rounded_up is the smallest power of 2 that is larger than n. This is also the number of 
                        columns in the tensor pointed to by terms_ptr.

    :param NUM_POINTS_TO_LOAD: hyperparameter. This specifies the number of points to evaluate monomials on at the same time.

    :param NUM_MONOMIALS_TO_LOAD: hyperparameter. This specifies the number of monomials to evaluate at the same time.

    :param dtype: Specifies whether the output will be float32 or float64.
    """
    num_monomials_indexer = tl.arange(0, NUM_MONOMIALS_TO_LOAD)[None, :, None]

    x_pid = tl.program_id(0)
    x_start = x_pid * NUM_POINTS_TO_LOAD * input_dimension_rounded_up
    x_offsets = x_start + tl.arange(
        0, NUM_POINTS_TO_LOAD * input_dimension_rounded_up
    )
    x_mask = x_offsets < num_points * input_dimension_rounded_up
    x = tl.load(input_ptr + x_offsets, mask=x_mask)
    x = x.reshape((NUM_POINTS_TO_LOAD, input_dimension_rounded_up))

    point_offsets = x_pid * NUM_POINTS_TO_LOAD + tl.arange(0, NUM_POINTS_TO_LOAD)
    point_mask = point_offsets < num_points
    output_offsets = point_offsets * num_polynomials

    which_batch = 0
    for poly in range(num_polynomials):
        polynomial_size = tl.load(polynomial_sizes_ptr + poly)
        output = tl.zeros((NUM_POINTS_TO_LOAD,), dtype=dtype)
        num_monomial_loops = tl.cdiv(polynomial_size, NUM_MONOMIALS_TO_LOAD)

        for monomial_idx in range(num_monomial_loops):
            coef_arange = tl.arange(0, NUM_MONOMIALS_TO_LOAD) + which_batch
            coef_mask = coef_arange < num_monomials
            coef = tl.load(coefs_ptr + coef_arange, mask=coef_mask)[None, :]
            coef = coef.broadcast_to(
                (NUM_POINTS_TO_LOAD, NUM_MONOMIALS_TO_LOAD)
            )

            terms_arange = (
                tl.arange(0, NUM_MONOMIALS_TO_LOAD * degree_rounded_up)
                + which_batch * degree_rounded_up
            )
            terms_mask = terms_arange < num_monomials * degree_rounded_up
            terms_index = tl.load(terms_ptr + terms_arange, mask=terms_mask)
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
                monomial_idx * NUM_MONOMIALS_TO_LOAD
                + num_monomials_indexer
                < polynomial_size,
                terms,
                0,
            )
            terms = tl.reduce(terms, 2, prod)
            terms *= coef
            terms = tl.sum(terms, axis=1)
            output += terms

            which_batch += min(
                polynomial_size - monomial_idx * NUM_MONOMIALS_TO_LOAD,
                NUM_MONOMIALS_TO_LOAD,
            )

        tl.store(output_ptr + output_offsets + poly, output, mask=point_mask)
