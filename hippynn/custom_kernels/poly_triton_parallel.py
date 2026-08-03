"""Polynomial-parallel Triton evaluator.

Unlike the original backend, the second launch dimension assigns one Triton
program to each polynomial, allowing different polynomials to run in parallel.
"""

import triton
import triton.language as tl


def evaluate_polynomials(
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
    evaluate_polynomials_kernel[grid](
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
    candidates = (
        (32, 16, 4, 2),
        (32, 32, 4, 2),
        (64, 16, 4, 2),
        (64, 32, 8, 2),
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
def evaluate_polynomials_kernel(
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
    """Evaluate one polynomial per program for a block of input points."""
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
