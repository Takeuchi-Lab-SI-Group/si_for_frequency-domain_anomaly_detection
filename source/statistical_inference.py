import numpy as np
from scipy.stats import chi
from sicore.core.base import SelectiveInference
from sicore.core.real_subset import RealSubset
from source import fourier
from source.changepoint_detection import ChangepointDetection
from source.si import HypothesisTesting


class SelectiveInferenceChi(SelectiveInference):
    def __init__(self, stat, degree, a, b):
        self.stat = stat
        self.a = a
        self.b = b
        degree = int(degree + 1e-3)
        self.mode = np.sqrt(degree - 1)
        self.support = RealSubset([[0.0, np.inf]])
        self.limits = (
            RealSubset([[self.mode - 20.0, np.max([self.stat, self.mode]) + 10.0]])
            & self.support
        )
        self.null_rv = chi(df=degree)
        self.alternative = "less"


class ChangepointInference(ChangepointDetection):
    def inference(self, tau, sigma_inference, parametric=True):
        if not hasattr(self, "F_freq"):
            raise ValueError("Run changepoint_detection() first.")

        self.tau = tau
        self.freq_tau = [
            d for i, d in enumerate(self.F_freq[1]) if self.tau in self.cp_list_sa[i]
        ]

        if len(self.freq_tau) == 0:
            # raise ValueError("tau is not a changepoint.")
            return None

        self.FFT_mat = fourier.calc_FFT_mat(self.window)

        testing = HypothesisTesting(
            self.tau,
            self.freq_tau,
            self.F_freq,
            sigma_inference,
            self.window,
            self.FFT_mat,
            self.cp_list_sa,
        )
        stat = testing.calc_test_statistic(self.x)
        self.FFT_mat_a, self.FFT_mat_b = testing.calc_FFT_mat_ab()

        calculator = SelectiveInferenceChi(stat, testing.dof, testing.a, testing.b)
        self.limits = calculator.limits

        result_OC = calculator.inference(
            self.algorithm,
            self.model_selector,
            inference_mode="over_conditioning",
        )

        if parametric:
            result_para = calculator.inference(
                self.algorithm,
                self.model_selector,
                inference_mode="parametric",
            )

            return result_OC.naive_p_value(), result_OC.p_value, result_para.p_value

        return result_OC.naive_p_value(), result_OC.p_value

    def algorithm(self, a, b, z):
        x = a + b * z
        CP_detection = ChangepointDetection(
            x,
            self.sigma,
            self.window,
            self.kappa,
            self.freq_l,
            self.freq_h,
            self.seed,
            a,
            b,
            self.FFT_mat_a,
            self.FFT_mat_b,
            self.freq_tau,
            SI_mode=True,
        )
        CP_detection.detect_changepoints()
        intervals = RealSubset(CP_detection.interval_ts) & self.limits
        return CP_detection.cp_list_sa, intervals

    def model_selector(self, CP_detection_cp_list_sa):
        return self.cp_list_sa == CP_detection_cp_list_sa
