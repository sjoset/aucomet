
import astropy.units as u
import numpy as np


# TODO: document this better
class TimeDependentProduction:
    """
        class for generating time-dependent production functions to hand to
        the vectorial model included in sbpy

        create the class by passing it a supported type, and get the time function
        by calling create_production
    """

    def __init__(self, type):
        self.supported_types = {'square pulse', 'gaussian', 'sine wave'}
        self.type = type

    def create_production(self, **kwargs):
        if self.type == "square pulse":
            allowed_keys = {'amplitude', 't_start', 'duration'}
            self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
            return self.make_square_pulse_q_t()
        elif self.type == "gaussian":
            allowed_keys = {'amplitude', 't_max', 'std_dev'}
            self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
            return self.make_gaussian_q_t()
        elif self.type == "sine wave":
            allowed_keys = {'amplitude', 'period', 'delta'}
            self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
            return self.make_sine_q_t()
        else:
            return None

    def make_square_pulse_q_t(self):
        t_start_in_secs = self.t_start.to(u.s).value
        tend_in_secs = (self.t_start - self.duration).to(u.s).value
        amplitude_in_invsecs = self.amplitude.to(1/(u.s)).value

        def q_t(t):
            # Comparisons seem backward because of our weird time system
            if t < t_start_in_secs and t > tend_in_secs:
                return amplitude_in_invsecs
            else:
                return 0

        return q_t

    def make_gaussian_q_t(self):
        t_max_in_secs = self.t_max.to(u.s).value
        std_dev_in_secs = self.std_dev.to(u.s).value
        amplitude_in_invsecs = self.amplitude.to(1/(u.s)).value

        def q_t(t):
            return amplitude_in_invsecs * np.e**-(((t - t_max_in_secs)**2)/(2*std_dev_in_secs**2))

        return q_t

    def make_sine_q_t(self):
        period_in_secs = self.period.to(u.s).value
        delta_in_secs = self.delta.to(u.s).value
        amplitude_in_invsecs = self.amplitude.to(1/(u.s)).value
        const_B = (2.0 * np.pi)/period_in_secs

        def q_t(t):
            return amplitude_in_invsecs * (
                    np.sin(const_B*(t + delta_in_secs)) + 1
                    )

        return q_t
