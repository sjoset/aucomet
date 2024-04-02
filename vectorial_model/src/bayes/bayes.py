import time
import emcee
import numpy as np


def lambdas_grid(**args):
    pass


def structure(**args):
    pass


def energy_grid(**args):
    pass


def interpolators(**args):
    pass


def lambda_distribution(
    ion,
    up_dir,
    x_bnd,
    x_res,
    nproc_str,
    nist_cutoff=0.05,
    n_lambdas=2,
    n_walkers=100,
    n_steps=1000,
    prior_shape="uniform",
    likelihood_shape="uniform",
    plot=True,
    outfile=None,
    cent_pot=1,
    emax=2.0,
):
    X_1D, x_ravel = lambdas_grid(x_bnd, x_res)

    # NIST data
    y_nist = structure(up_dir, ion, lambdas=[])[1]

    # Construct interpolators
    n_energies = y_nist.size

    start_egrid = time.time()
    Err, Erg = energy_grid(ion, up_dir, x_ravel, x_res, cent_pot, emax, nproc_str)

    start_interp = time.time()
    print("Time in Egrid=", (start_interp - start_egrid) / 60.0, " minutes")
    err_interpolators = interpolators(X_1D, Err)

    # =============================================================================
    # Run MCMC Code
    # =============================================================================
    #
    # Likelihood is based on error*100% error in each component
    #
    y_bnd = np.zeros((n_energies, 2))
    # Note I replaced n_lambdas here with just 2; I think it just gives thr bounds on each energy in the MCMC search.
    #    y_bnd = np.zeros((n_energies, n_lambdas))
    for i in range(n_energies):
        y_bnd[i, :] = -1, 1

    y_bnd *= nist_cutoff

    #
    # Specify starting points for each Markov chain (in a tight ball around optimum)
    #
    pos = [
        np.array([1 for i in range(n_lambdas)]) + 1e-4 * np.random.randn(n_lambdas)
        for i in range(n_walkers)
    ]

    #
    # Initialize the sampler
    #
    start_sampler = time.time()
    print(
        "Time in make interpolators=", (start_sampler - start_interp) / 60.0, " minutes"
    )
    if nproc_str > 1:
        pool = mp.Pool(nproc_str, maxtasksperchild=1)
        with mp.Pool() as pool:
            sampler = emcee.EnsembleSampler(
                n_walkers,
                n_lambdas,
                log_posterior,
                pool=pool,
                args=(err_interpolators, x_bnd, y_bnd, prior_shape, likelihood_shape),
            )
            sampler.run_mcmc(pos, n_steps)
    else:
        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_lambdas,
            log_posterior,
            args=(err_interpolators, x_bnd, y_bnd, prior_shape, likelihood_shape),
        )
        sampler.run_mcmc(pos, n_steps)

    finish_mcmc = time.time()
    print("Time in Emcee MCMC=", (finish_mcmc - start_sampler) / 60.0, " minutes")
    #    print('Completed MCMC calculation:')
    acceptance = np.mean(sampler.acceptance_fraction)
    #    print("Mean acceptance fraction: {0:.3f}".format(acceptance))
    autocorrel = np.mean(sampler.get_autocorr_time())
    #    print("Mean autocorrelation time: {0:.3f} steps".format(autocorrel))
    #    acceptance=0.8
    #    autocorrel=0.65
    finish_lambdas = time.time()
    print(
        "Time in autocorrel and acceptance=",
        (finish_lambdas - finish_mcmc) / 60.0,
        " minutes",
    )
    #
    # The sampler.chain has shape (n_walkers, n_steps, n_dim)
    # Reshape array of samples (but throw away initial sample)
    #
    lambda_samples = sampler.chain[:, 50:, :].reshape((-1, n_lambdas))

    #    if outfile is not None:
    #        np.save(outfile, arr=lambda_samples,allow_pickle=True)

    return lambda_samples, Err, Erg, err_interpolators, y_nist, acceptance, autocorrel
