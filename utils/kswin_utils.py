from collections import deque
import random

from scipy.stats import ks_2samp

class KSWINDetector:
    def __init__(self,
                 reference_window_size=1000,
                 current_window_size=336,
                 alpha = 0.005, #balanced choice between sensitivity and false positive rate
                 random_state=42
    ):
        self.reference_window_size = reference_window_size
        self.current_window_size = current_window_size
        self.alpha = alpha
    
        self.window_size = (
            self.reference_window_size + 
            self.current_window_size
        )
        self.window = deque(maxlen=self.window_size)
        self.random = random.Random(random_state)
    
    def is_ready(self):
        return len(self.window) == self.window_size
    
    def current_size(self):
        return len(self.window)
    
    def detect_drift(self):
        if not self.is_ready():
            return {
                "drift" : False,
                "p_value" : None,
                "ks_statistic" : None
            }
        
        window_list = list(self.window)
        reference_window = window_list[:self.reference_window_size]
        current_window = window_list[self.reference_window_size:]
        
        sampled_reference = self.random.sample(
            reference_window,
            self.current_window_size
        )
        
        print(reference_window[:5])
        print(current_window[:5])
        
        ks_statistic, p_value = ks_2samp(sampled_reference, current_window)
        
        return {
            "drift" : p_value < self.alpha,
            "p_value" : p_value,
            "ks_statistic" : ks_statistic
        }
    
    def update(self, value):

        self.window.append(value)

        result = self.detect_drift()
        # Once the drift is detected, reset the window by discarding older observations
        if result["drift"]:

            window_list = list(self.window)

            current_window = window_list[self.reference_window_size:]

            self.window.clear()

            self.window.extend(current_window)

        return result