Speed up the experimental MJMPM solver's rigid contact solve by sweeping each world on its own thread with precomputed contact constants, preserving the sequential per-world contact order.
