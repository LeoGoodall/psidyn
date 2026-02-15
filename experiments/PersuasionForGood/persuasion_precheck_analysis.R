library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

results_dir <- "PersuasionForGood/results/prechecks"
lag_summary_path <- file.path(results_dir, "te_lag_scan.csv")

if (file.exists(lag_summary_path)) {
  lag_summary_df <- read_csv(lag_summary_path, show_col_types = FALSE) %>%
    arrange(lag)
  required_cols <- c(
    "lag",
    "median_total_te",
    "surrogate_median_total_te",
    "std_total_te",
    "std_total_te_surrogate"
  )
  if (all(required_cols %in% names(lag_summary_df))) {
    max_corrected <- max(lag_summary_df$median_total_te_minus_surrogate, na.rm = TRUE)
    lag_summary_df <- lag_summary_df %>%
      mutate(
        corrected_te = median_total_te_minus_surrogate,
        snr = if_else(std_total_te > 0, corrected_te / std_total_te, NA_real_),
        marginal_gain_corrected = corrected_te - lag(corrected_te),
        marginal_gain_pct = marginal_gain_corrected / lag(corrected_te),
        corrected_te_pct_of_max = if (max_corrected > 0) {
          corrected_te / max_corrected
        } else {
          NA_real_
        }
      ) %>%
      mutate(
        marginal_gain_corrected = coalesce(marginal_gain_corrected, corrected_te),
        marginal_gain_pct = coalesce(marginal_gain_pct, 1)
      )

    loess_fit <- loess(corrected_te ~ lag, data = lag_summary_df, span = 0.4)
    lag_summary_df <- lag_summary_df %>%
      mutate(corrected_te_smoothed = predict(loess_fit, lag))

    max_smoothed <- max(lag_summary_df$corrected_te_smoothed, na.rm = TRUE)
    elbow_threshold <- 0.95 * max_smoothed
    elbow_lag <- lag_summary_df %>%
      filter(!is.na(corrected_te_smoothed), corrected_te_smoothed >= elbow_threshold) %>%
      summarise(elbow = min(lag, na.rm = TRUE)) %>%
      pull(elbow)
    if (length(elbow_lag) == 0 || is.na(elbow_lag)) {
      elbow_lag <- lag_summary_df %>%
        filter(corrected_te == max(corrected_te, na.rm = TRUE)) %>%
        summarise(elbow = min(lag, na.rm = TRUE)) %>%
        pull(elbow)
    }
    lag_summary_df <- lag_summary_df %>%
      mutate(elbow_flag = lag == elbow_lag)
    optimal_lag <- elbow_lag

    write_csv(lag_summary_df, lag_summary_path)

    lag_long <- bind_rows(
      lag_summary_df %>%
        transmute(
          lag,
          stat_type = "Median total TE",
          series = "Observed",
          value = median_total_te
        ),
      lag_summary_df %>%
        transmute(
          lag,
          stat_type = "Median total TE",
          series = "Surrogate",
          value = surrogate_median_total_te
        ),
      lag_summary_df %>%
        transmute(
          lag,
          stat_type = "Std total TE",
          series = "Observed",
          value = std_total_te
        ),
      lag_summary_df %>%
        transmute(
          lag,
          stat_type = "Std total TE",
          series = "Surrogate",
          value = std_total_te_surrogate
        ),
      lag_summary_df %>%
        transmute(
          lag,
          stat_type = "Corrected TE (Observed - Surrogate)",
          series = "Corrected",
          value = corrected_te
        )
    ) %>%
      mutate(
        stat_type = factor(
          stat_type,
          levels = c(
            "Median total TE",
            "Corrected TE (Observed - Surrogate)",
            "Std total TE"
          )
        )
      )

    lag_plot <- ggplot(lag_long, aes(x = lag, y = value, colour = series)) +
      geom_line(na.rm = TRUE) +
      geom_point(na.rm = TRUE) +
      facet_wrap(~stat_type, ncol = 1, scales = "free_y") +
      geom_vline(xintercept = optimal_lag, colour = "orange", linetype = "dashed") +
      geom_point(
        data = lag_long %>% filter(lag == optimal_lag, series %in% c("Observed", "Corrected")),
        size = 2.5
      ) +
      labs(
        x = "Lag window",
        y = "Transfer entropy (bits)",
        title = "Lag scan summary: cumulative vs corrected TE",
        colour = "Series"
      ) +
      scale_colour_manual(
        values = c(
          "Observed" = "#1b9e77",
          "Surrogate" = "#d95f02",
          "Corrected" = "#7570b3"
        )
      ) +
      theme_minimal()

    combined_plot_path <- file.path(results_dir, "lag_selection_median_std_te.png")
    ggsave(combined_plot_path, plot = lag_plot, width = 7, height = 7.5, dpi = 300)

  } else {
    warning("Lag summary CSV missing required columns: ", lag_summary_path)
  }
} else {
  warning("Lag summary CSV not found at: ", lag_summary_path)
}
