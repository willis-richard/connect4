l_total, l_correct, d_total, d_correct, w_total, w_correct = 33874, 2395, 12887, 2812, 115986, 69352

max_possible_error_squared = \
        (l_total - l_correct) * 1.0**2 + \
        l_correct * 0.333**2 + \
        (d_total - d_correct) * 0.5**2 + \
        d_correct * (0.333 / 2.0)**2 + \
        (w_total - w_correct) * 1.0**2 + \
        w_correct * 0.333**2

min_possible_error_squared = \
        (l_total - l_correct) * 0.333**2 + \
        (d_total - d_correct) * (0.333 / 2.0)**2 + \
        (w_total - w_correct) * 0.333**2

MaxSE = max_possible_error_squared / (l_total + d_total + w_total)
MinSE = min_possible_error_squared / (l_total + d_total + w_total)

print(MaxSE, MinSE)

