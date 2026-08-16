# Leakage audit

Generated from `src/evaluation/leakage.py`. The mutation test is enforced in
`tests/test_leakage.py`: all observations after a cutoff are perturbed, features
are rebuilt, and every feature value at or before the cutoff must be unchanged.

| Feature | Definition | Historical cutoff | Leakage risk | Leakage test |
|---|---|---|---|---|
| price | current observed price at T | <= T | none | mutation_test |
| price_{d}d_ago | closest observation >= d days before T | <= T | none | mutation_test |
| return_{d}d | price / price_{d}d_ago - 1 | <= T | none | mutation_test |
| volatility_{d}d | annualized std of daily log returns over trailing d days | <= T | none | mutation_test |
| historical_high/low | expanding max/min of past prices | <= T | none | mutation_test |
| distance_from_high/low | price relative to expanding extremes | <= T | none | mutation_test |
| price_momentum/acceleration | trailing 30d return and its change | <= T | none | mutation_test |
| obs_count_90d, days_since_last_obs, obs_number | trailing observation bookkeeping | <= T | none | mutation_test |
| market_* | cross-card medians merged as-of T (backward) | <= T | low: verify merge direction | mutation_test |
| card_age_*, is_rookie_card | static metadata + T - release year | static | none | n/a |
| prev_season_* | latest season with season_end_date <= T (merge_asof backward) | season end <= T | low: verify season_end_date is correct | mutation_test |
| future_price_{h}d / future_return_{h}d | TARGETS: only compared against, never features | > T by design | must never enter feature list | feature-list check in tests |

## Forbidden by construction

- Future prices (targets are never in the feature list — asserted in tests)
- Future player statistics (season joins are backward-only on `season_end_date`)
- Future market averages (market index merged with `direction="backward"`)
- Future normalization statistics (imputers/scalers are fitted inside each fold on training rows only)
- Future population reports and future sales (not in the dataset at all)
