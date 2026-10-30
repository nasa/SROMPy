# Copyright 2018 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in
# the United States under Title 17, U.S. Code. All Other Rights Reserved.

# The Stochastic Reduced Order Models with Python (SROMPy) platform is licensed
# under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0.

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""
End-to-end tests of SROM.optimize and of the objective it minimizes, on
targets whose ranges are not [0, 1] (the spring-mass inputs of the 2D example).
"""

import numpy as np
import pytest

from SROMPy.optimize.Gradient import Gradient
from SROMPy.optimize.ObjectiveFunction import ObjectiveFunction
from SROMPy.optimize.Optimizer import Optimizer
from SROMPy.srom import SROM
from SROMPy.target import BetaRandomVariable, NormalRandomVariable, \
    SampleRandomVector, UniformRandomVariable
from SROMPy.target.RandomVector import RandomVector


@pytest.fixture
def spring_mass_samples():
    """Stiffness on [1, 3.5] and mass on [0.5, 2], as in the 2D example."""
    np.random.seed(0)
    stiffness = BetaRandomVariable(alpha=3., beta=2., shift=1., scale=2.5)
    mass = BetaRandomVariable(alpha=2. / 3., beta=1. / 3., shift=.5, scale=1.5)
    return SampleRandomVector(np.array([stiffness.draw_random_sample(2000),
                                        mass.draw_random_sample(2000)]).T)


def test_random_variables_have_dim():
    for variable in (BetaRandomVariable(alpha=3., beta=2.),
                     NormalRandomVariable(),
                     UniformRandomVariable(0., 1.)):
        assert variable.dim == 1


def test_sequential_objective_uses_samples_unchanged(spring_mass_samples):
    """Sequential optimization fixes the samples: neither the objective nor
    its gradient may move them (they were clipped into the probabilities'
    bounds, [0.01, 0.99])."""
    srom = SROM(10, 2)
    samples = spring_mass_samples.draw_random_sample(10)
    probabilities = np.full(10, .1)
    objective = ObjectiveFunction(srom, spring_mass_samples, error="SSE")
    value = objective.evaluate(samples, probabilities)
    np.testing.assert_array_equal(srom.samples, samples)

    reference = SROM(10, 2)
    reference.set_params(samples, probabilities)
    by_parts = ObjectiveFunction(reference, spring_mass_samples, error="SSE")
    expected = by_parts.compute_cdf_error() + by_parts.compute_moment_error() + \
        by_parts.compute_correlation_error()
    assert value == pytest.approx(expected)

    gradient = Gradient(srom, spring_mass_samples, error="SSE")
    gradient.evaluate(samples, probabilities)
    np.testing.assert_array_equal(srom.samples, samples)


def test_joint_objective_bounds_each_coordinate(spring_mass_samples):
    """Joint optimization keeps each coordinate inside its own range."""
    srom = SROM(3, 2)
    samples = np.array([[2., 1.], [9., 1.], [2., -4.]])
    objective = ObjectiveFunction(srom, spring_mass_samples, error="SSE",
                                  joint_opt=True)
    objective.evaluate(samples.copy(), np.full(3, 1. / 3.))
    mins, maxs = spring_mass_samples.mins, spring_mass_samples.maxs
    assert np.all(srom.samples > mins) and np.all(srom.samples < maxs)
    np.testing.assert_array_equal(srom.samples[0], samples[0])
    assert srom.samples[1, 1] == samples[1, 1]
    assert srom.samples[2, 0] == samples[2, 0]


class _Stub(RandomVector):
    """A 2D target with given CDF values and moments."""

    def __init__(self, cdfs, moments):
        super(_Stub, self).__init__(2)
        self._cdfs, self._moments = cdfs, moments
        self.mins, self.maxs = [0., 0.], [1., 1.]

    def compute_cdf(self, x_grid):
        return self._cdfs.copy()

    def compute_moments(self, max_order):
        return self._moments.copy()

    def compute_correlation_matrix(self):
        return np.eye(2)

    def draw_random_sample(self, sample_size):
        return np.random.random((sample_size, 2))


def test_cdf_error_skips_zeros_of_each_input():
    """A zero target CDF in the second input, where the first is positive,
    is skipped rather than divided by."""
    cdfs = np.array([[0., 0.], [.5, 0.], [1., .5], [1., 1.]])
    srom = SROM(2, 2)
    srom.set_params(np.array([[.2, .7], [.6, .9]]), np.array([.5, .5]))
    objective = ObjectiveFunction(srom, _Stub(cdfs, np.ones((5, 2))),
                                  error="SSE", num_cdf_grid_points=4)
    srom_cdfs = srom.compute_cdf(objective._x_grid)
    kept = cdfs > 0
    expected = .5 * np.sum(((srom_cdfs - cdfs)[kept] / cdfs[kept]) ** 2)
    with np.errstate(divide="raise", invalid="raise"):
        assert objective.compute_cdf_error() == pytest.approx(expected)


def test_moment_error_guards_each_input():
    """A zero target moment in one input does not replace the other input's
    moment of the same order."""
    moments = np.array([[0., 2.], [1., 5.]])
    srom = SROM(2, 2)
    srom.set_params(np.array([[-1., 1.], [1., 2.]]), np.array([.5, .5]))
    objective = ObjectiveFunction(srom, _Stub(np.ones((4, 2)), moments),
                                  error="SSE", max_moment=2)
    srom_moments = srom.compute_moments(2)
    guarded = np.where(moments == 0., 1., moments)
    expected = .5 * np.sum(((srom_moments - moments) / guarded) ** 2)
    assert objective.compute_moment_error() == pytest.approx(expected)


def test_initial_guess_belongs_to_the_samples(spring_mass_samples):
    """The sequential initial guess is the share of the target's samples
    nearest to each of the SROM samples being optimized."""
    optimizer = Optimizer(spring_mass_samples, SROM(5, 2))
    samples = spring_mass_samples.draw_random_sample(5)
    guess = optimizer.get_initial_guess(False, samples=samples)
    target = spring_mass_samples.samples
    nearest = np.argmin(np.linalg.norm(target[:, None, :] - samples[None, :, :],
                                       axis=2), axis=1)
    np.testing.assert_allclose(guess, np.bincount(nearest, minlength=5) /
                               len(target))


def test_initial_guess_without_target_samples():
    """An analytic target has no samples: equal probabilities, as before."""
    optimizer = Optimizer(BetaRandomVariable(alpha=3., beta=2.), SROM(4, 1))
    np.testing.assert_allclose(optimizer.get_initial_guess(False), np.full(4, .25))


def test_best_result_kept_above_1e6(spring_mass_samples):
    """The best trial is kept however large its objective."""
    srom = SROM(5, 2)
    optimizer = Optimizer(spring_mass_samples, srom, obj_weights=[1e9, 1e9, 1e9])
    samples, probabilities = optimizer.get_optimal_params(
        num_test_samples=2, verbose=False)
    assert samples is not None and probabilities is not None


def test_joint_optimization_needs_scale(spring_mass_samples):
    with pytest.raises(ValueError):
        Optimizer(spring_mass_samples, SROM(5, 2), joint_opt=True)


@pytest.mark.parametrize("joint_opt", [False, True])
@pytest.mark.parametrize("case", ["beta", "spring_mass"])
def test_optimize_end_to_end(case, joint_opt, spring_mass_samples):
    """SROM.optimize returns valid samples and probabilities that improve on
    equal probabilities at the returned samples."""
    if case == "beta":
        target, size, dim = BetaRandomVariable(alpha=3., beta=2., shift=1.,
                                               scale=2.5), 6, 1
    else:
        target, size, dim = spring_mass_samples, 6, 2
    np.random.seed(1)
    srom = SROM(size, dim)
    srom.optimize(target, num_test_samples=3, joint_opt=joint_opt,
                  verbose=False, cdf_grid_pts=20,
                  scale=.1 if joint_opt else None)
    samples, probabilities = srom.get_params()
    samples = samples.reshape(size, dim)
    assert np.all(np.isfinite(samples)) and probabilities.sum() == pytest.approx(1.)
    assert np.all(samples >= np.array(target.mins) - 1e-12)
    assert np.all(samples <= np.array(target.maxs) + 1e-12)

    def objective(p):
        check = SROM(size, dim)
        check.set_params(samples, p)
        return ObjectiveFunction(check, target, error="SSE",
                                 num_cdf_grid_points=20).evaluate(samples, p)

    assert objective(probabilities) <= objective(np.full(size, 1. / size))


def test_sequential_gradient_matches_finite_differences(spring_mass_samples):
    """The objective is quadratic in the probabilities for fixed samples, so
    central differences agree with the gradient to rounding."""
    np.random.seed(2)
    samples = spring_mass_samples.draw_random_sample(8)
    probabilities = np.random.dirichlet(np.ones(8))
    objective = ObjectiveFunction(SROM(8, 2), spring_mass_samples, error="SSE",
                                  num_cdf_grid_points=50)
    gradient = Gradient(SROM(8, 2), spring_mass_samples, error="SSE",
                        cdf_grid_pts=50)
    step = 1e-6
    central = np.array([(objective.evaluate(samples, probabilities + step * e) -
                         objective.evaluate(samples, probabilities - step * e)) /
                        (2 * step) for e in np.eye(8)])
    np.testing.assert_allclose(gradient.evaluate(samples, probabilities), central,
                               rtol=1e-6)
