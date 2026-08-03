"""Original Triton polynomial evaluator.

This backend intentionally preserves the one-dimensional launch and serial
loop over polynomials used before the polynomial-parallel optimization.
"""

import triton
import triton.language as tl


def evaluate_polynomials(
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
    """Launch the original serial-over-polynomials kernel."""
    grid = lambda meta: (
        triton.cdiv(num_points, meta["NUM_POINTS_TO_LOAD"]),
    )
    evaluate_polynomials_kernel[grid](
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
    """Return the original kernel's autotuning configuration."""
    configs = []
    for num_points_to_load in [128, 256, 512]:
        for num_monomials_to_load in [8, 16, 32]:
            for num_warps in [2, 4, 8, 16, 32]:
                for num_stages in [2, 3, 4, 5, 6]:
                    configs.append(
                        triton.Config(
                            kwargs={
                                "NUM_POINTS_TO_LOAD": num_points_to_load,
                                "NUM_MONOMIALS_TO_LOAD": num_monomials_to_load,
                            },
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs


@triton.jit
def prod(x, y):
    return x * y


@triton.autotune(configs=get_configs_polynomials(), key=["num_monomials"])
@triton.jit
def evaluate_polynomials_kernel(
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
    """Evaluate every polynomial serially within each point-block program."""
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
