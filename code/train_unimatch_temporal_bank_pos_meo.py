"""Temporal Bank v2 refinement with full POS/MEO gradient composition.

The pseudo-label bank, data split, augmentations, routing thresholds,
curriculum, optimizer, learning-rate schedule, EMA, validation, and
checkpoint logic are inherited unchanged from Temporal Bank v2.  Only the
composition of the supervised and weighted bank-loss gradients is replaced.

POS solves the two-objective minimum-norm convex combination:

    min ||alpha_s * g_s + alpha_u * g_u||^2
    subject to alpha_s + alpha_u = 1 and alpha_s, alpha_u >= 0.

MEO then restores the norm of the equal-weight reference gradient:

    h = normalize(g_POS) * ||0.5 * g_s + 0.5 * g_u||.

Here g_u is the gradient of consistency_weight * L_bank, preserving the
original v2 ramp-up and maximum consistency weight.
"""

import logging

import torch

import train_unimatch_temporal_bank_v2 as temporal_v2


_step_count = 0
_conflict_count = 0
_announced = False


def pos_meo_optimizer_step(
    model,
    optimizer,
    supervised_loss,
    unsupervised_loss,
    consistency_weight,
    epsilon=1e-12,
):
    """Apply the analytic two-task POS solution followed by MEO."""
    global _announced, _conflict_count, _step_count

    if not _announced:
        logging.info(
            "Gradient composition: full POS + MEO; "
            "g_unsup = grad(consistency_weight * bank_loss)"
        )
        _announced = True

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    weighted_unsupervised_loss = (
        float(consistency_weight) * unsupervised_loss
    )
    total_loss = supervised_loss + weighted_unsupervised_loss

    optimizer.zero_grad()
    supervised_gradients = torch.autograd.grad(
        supervised_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    unsupervised_gradients = torch.autograd.grad(
        weighted_unsupervised_loss,
        parameters,
        retain_graph=False,
        allow_unused=True,
    )

    scalar_options = {
        "device": supervised_loss.device,
        "dtype": supervised_loss.dtype,
    }
    norm_supervised_sq = torch.zeros((), **scalar_options)
    norm_unsupervised_sq = torch.zeros((), **scalar_options)
    inner_product = torch.zeros((), **scalar_options)
    for supervised_gradient, unsupervised_gradient in zip(
        supervised_gradients, unsupervised_gradients
    ):
        if supervised_gradient is not None:
            norm_supervised_sq.add_(
                torch.sum(supervised_gradient * supervised_gradient)
            )
        if unsupervised_gradient is not None:
            norm_unsupervised_sq.add_(
                torch.sum(unsupervised_gradient * unsupervised_gradient)
            )
        if (
            supervised_gradient is not None
            and unsupervised_gradient is not None
        ):
            inner_product.add_(
                torch.sum(supervised_gradient * unsupervised_gradient)
            )

    denominator = (
        norm_supervised_sq
        + norm_unsupervised_sq
        - 2.0 * inner_product
    )
    if float(denominator.detach()) > float(epsilon):
        alpha_unsupervised = (
            (norm_supervised_sq - inner_product) / denominator
        ).clamp(0.0, 1.0)
    else:
        alpha_unsupervised = torch.full(
            (), 0.5, **scalar_options
        )
    alpha_supervised = 1.0 - alpha_unsupervised

    norm_pos_sq = torch.zeros((), **scalar_options)
    norm_uniform_sq = torch.zeros((), **scalar_options)
    for parameter, supervised_gradient, unsupervised_gradient in zip(
        parameters, supervised_gradients, unsupervised_gradients
    ):
        if supervised_gradient is None and unsupervised_gradient is None:
            continue
        if supervised_gradient is None:
            supervised_gradient = torch.zeros_like(parameter)
        if unsupervised_gradient is None:
            unsupervised_gradient = torch.zeros_like(parameter)
        pos_gradient = (
            alpha_supervised * supervised_gradient
            + alpha_unsupervised * unsupervised_gradient
        )
        uniform_gradient = 0.5 * (
            supervised_gradient + unsupervised_gradient
        )
        norm_pos_sq.add_(torch.sum(pos_gradient * pos_gradient))
        norm_uniform_sq.add_(torch.sum(uniform_gradient * uniform_gradient))

    norm_pos = torch.sqrt(norm_pos_sq.clamp_min(0.0))
    norm_uniform = torch.sqrt(norm_uniform_sq.clamp_min(0.0))
    degenerate = float(norm_pos.detach()) <= float(epsilon)
    if degenerate:
        # The MEO direction is undefined only for an exact zero POS vector.
        # The equal-weight reference is the finite limiting fallback.
        meo_scale = torch.ones((), **scalar_options)
    else:
        meo_scale = norm_uniform / norm_pos.clamp_min(float(epsilon))

    for parameter, supervised_gradient, unsupervised_gradient in zip(
        parameters, supervised_gradients, unsupervised_gradients
    ):
        if supervised_gradient is None and unsupervised_gradient is None:
            parameter.grad = None
            continue
        if supervised_gradient is None:
            supervised_gradient = torch.zeros_like(parameter)
        if unsupervised_gradient is None:
            unsupervised_gradient = torch.zeros_like(parameter)
        if degenerate:
            final_gradient = 0.5 * (
                supervised_gradient + unsupervised_gradient
            )
        else:
            final_gradient = meo_scale * (
                alpha_supervised * supervised_gradient
                + alpha_unsupervised * unsupervised_gradient
            )
        parameter.grad = final_gradient.detach()

    optimizer.step()

    epsilon_tensor = torch.full((), float(epsilon), **scalar_options)
    norm_supervised = torch.sqrt(norm_supervised_sq.clamp_min(0.0))
    norm_unsupervised = torch.sqrt(norm_unsupervised_sq.clamp_min(0.0))
    cosine_denominator = (
        norm_supervised * norm_unsupervised
    ).clamp_min(epsilon_tensor)
    gradient_cosine = (
        inner_product / cosine_denominator
    ).clamp(-1.0, 1.0)

    _step_count += 1
    is_conflict = float(inner_product.detach()) < 0.0
    _conflict_count += int(is_conflict)
    norm_final = norm_uniform

    stats = {
        "alpha_supervised": float(alpha_supervised.detach()),
        "alpha_unsupervised": float(alpha_unsupervised.detach()),
        "gradient_cosine": float(gradient_cosine.detach()),
        "conflict": float(is_conflict),
        "conflict_rate": float(_conflict_count) / float(_step_count),
        "norm_supervised": float(norm_supervised.detach()),
        "norm_unsupervised": float(norm_unsupervised.detach()),
        "norm_uniform": float(norm_uniform.detach()),
        "norm_pos": float(norm_pos.detach()),
        "norm_final": float(norm_final.detach()),
        "meo_scale": float(meo_scale.detach()),
        "degenerate_fallback": float(degenerate),
    }
    return total_loss, stats


def build_parser():
    parser = temporal_v2.build_parser()
    parser.description = (
        "Temporal Bank v2 with full POS/MEO gradient composition"
    )
    parser.add_argument(
        "--pos_epsilon",
        type=float,
        default=1e-12,
        help="Numerical epsilon; it does not alter the POS/MEO objective.",
    )
    return parser


def main(args):
    def configured_step(
        model,
        optimizer,
        supervised_loss,
        unsupervised_loss,
        consistency_weight,
    ):
        return pos_meo_optimizer_step(
            model,
            optimizer,
            supervised_loss,
            unsupervised_loss,
            consistency_weight,
            epsilon=args.pos_epsilon,
        )

    temporal_v2.refinement_optimizer_step = configured_step
    temporal_v2.main(args)


if __name__ == "__main__":
    main(build_parser().parse_args())
