# AAE Corpus PID Analysis — Streamlined for Methods Paper
#
# Five essential analyses:
# 1. Descriptive statistics of PID atoms
# 2. One-sample t-test: is synergy different from zero?
# 3. Permutation null model: real vs shuffled
# 4. CCS vs MMI comparison
# 5. Semantic similarity correlations (if available)
#
# One composite figure: aae_pid_figure.pdf
#
# Input: pid_aae_results.csv (or pid_aae_results_with_similarity.csv), pid_aae_null_results.csv
# Output: Console + aae_statistical_analysis.json + aae_pid_figure.pdf

library(tidyverse)
library(effectsize)
library(jsonlite)
library(patchwork)

# ── Configuration ─────────────────────────────────────────────────────
RESULTS_FILE <- "pid_aae_results_with_similarity.csv"
NULL_RESULTS_FILE <- "pid_aae_null_results.csv"
OUTPUT_FILE <- "aae_statistical_analysis.json"
FIGURE_FILE <- "aae_pid_figure.pdf"

# ── Load data ─────────────────────────────────────────────────────────
load_data <- function() {
  cat("Loading PID results...\n")
  df <- read_csv(RESULTS_FILE, show_col_types = FALSE) %>%
    filter(token_count > 0, !is.na(synergy))
  cat(sprintf("  Valid samples: %d\n", nrow(df)))
  return(df)
}

section <- function(title) {
  cat("\n", strrep("=", 70), "\n")
  cat(title, "\n")
  cat(strrep("=", 70), "\n")
}

# ======================================================================
# 1. DESCRIPTIVE STATISTICS
# ======================================================================
descriptive_statistics <- function(df) {
  section("1. DESCRIPTIVE STATISTICS")

  cat(sprintf("\n  n = %d samples from %d essays\n", nrow(df), n_distinct(df$essay_id)))

  cat("\n  MMI decomposition (bits per token):\n")
  for (col in c("redundancy", "unique_x1", "unique_x2", "synergy")) {
    vals <- df[[col]]
    cat(sprintf("    %-12s  mean = %7.4f  SD = %.4f  median = %7.4f\n",
                col, mean(vals), sd(vals), median(vals)))
  }

  cat("\n  CCS decomposition (bits per token):\n")
  for (col in c("ccs_redundancy", "ccs_unique_x1", "ccs_unique_x2", "ccs_synergy")) {
    vals <- df[[col]]
    cat(sprintf("    %-16s  mean = %7.4f  SD = %.4f\n",
                col, mean(vals), sd(vals)))
  }

  cat("\n  Text lengths (characters):\n")
  cat(sprintf("    Claim:     mean = %.0f (SD = %.0f)\n", mean(df$claim_len), sd(df$claim_len)))
  cat(sprintf("    Premise 1: mean = %.0f (SD = %.0f)\n", mean(df$premise1_len), sd(df$premise1_len)))
  cat(sprintf("    Premise 2: mean = %.0f (SD = %.0f)\n", mean(df$premise2_len), sd(df$premise2_len)))

  # Brief note on asymmetry
  cat(sprintf("\n  Unique info asymmetry: Premise 1 (%.3f) > Premise 2 (%.3f),",
              mean(df$unique_x1), mean(df$unique_x2)))
  cat(sprintf(" consistent with Premise 1 being longer on average (%.0f vs %.0f chars)\n",
              mean(df$premise1_len), mean(df$premise2_len)))

  # Synergy ~ claim length (brief)
  r <- cor.test(df$synergy, df$claim_len)
  cat(sprintf("\n  Synergy ~ claim length: r = %.3f, p = %.4f\n", r$estimate, r$p.value))

  return(list(
    n = nrow(df),
    n_essays = n_distinct(df$essay_id),
    mmi = list(
      redundancy = list(mean = mean(df$redundancy), sd = sd(df$redundancy)),
      unique_x1 = list(mean = mean(df$unique_x1), sd = sd(df$unique_x1)),
      unique_x2 = list(mean = mean(df$unique_x2), sd = sd(df$unique_x2)),
      synergy = list(mean = mean(df$synergy), sd = sd(df$synergy))
    ),
    synergy_claim_len_r = as.numeric(r$estimate),
    synergy_claim_len_p = r$p.value
  ))
}

# ======================================================================
# 2. ONE-SAMPLE T-TEST ON SYNERGY
# ======================================================================
test_synergy_nonzero <- function(df) {
  section("2. ONE-SAMPLE T-TEST: IS SYNERGY DIFFERENT FROM ZERO?")

  t_res <- t.test(df$synergy, mu = 0)
  d <- cohens_d(df$synergy, mu = 0)

  cat(sprintf("\n  Mean synergy = %.4f (SD = %.4f)\n", mean(df$synergy), sd(df$synergy)))
  cat(sprintf("  t(%d) = %.4f, p = %.2e\n", t_res$parameter, t_res$statistic, t_res$p.value))
  cat(sprintf("  95%% CI: [%.4f, %.4f]\n", t_res$conf.int[1], t_res$conf.int[2]))
  cat(sprintf("  Cohen's d = %.4f\n", d$Cohens_d))

  direction <- ifelse(mean(df$synergy) < 0, "negative (sub-additive)", "positive (super-additive)")
  sig <- ifelse(t_res$p.value < 0.05, "SIGNIFICANT", "NOT SIGNIFICANT")
  cat(sprintf("\n  Result: %s — synergy is %s\n", sig, direction))

  return(list(
    mean = mean(df$synergy), sd = sd(df$synergy),
    t = as.numeric(t_res$statistic), df = as.numeric(t_res$parameter),
    p = t_res$p.value, cohens_d = d$Cohens_d,
    ci = c(t_res$conf.int[1], t_res$conf.int[2])
  ))
}

# ======================================================================
# 3. PERMUTATION NULL MODEL
# ======================================================================
test_null_model <- function(df) {
  section("3. PERMUTATION NULL MODEL: REAL VS SHUFFLED")

  if (!file.exists(NULL_RESULTS_FILE)) {
    cat("\n  WARNING: Null results file not found. Skipping.\n")
    return(list(skipped = TRUE))
  }

  df_null <- read_csv(NULL_RESULTS_FILE, show_col_types = FALSE) %>%
    filter(token_count > 0, !is.na(synergy))

  cat(sprintf("\n  Real samples: %d, Null samples: %d\n", nrow(df), nrow(df_null)))

  atoms <- c("synergy", "redundancy", "unique_x1", "unique_x2")
  atom_results <- list()

  for (atom in atoms) {
    t_res <- t.test(df[[atom]], df_null[[atom]])
    d_res <- cohens_d(
      c(df[[atom]], df_null[[atom]]),
      factor(c(rep("real", nrow(df)), rep("null", nrow(df_null))))
    )

    cat(sprintf("\n  %s:\n", atom))
    cat(sprintf("    Real: mean = %.4f (SD = %.4f)\n", mean(df[[atom]]), sd(df[[atom]])))
    cat(sprintf("    Null: mean = %.4f (SD = %.4f)\n", mean(df_null[[atom]]), sd(df_null[[atom]])))
    cat(sprintf("    t = %.4f, p = %.2e, d = %.4f\n",
                t_res$statistic, t_res$p.value, d_res$Cohens_d))

    atom_results[[atom]] <- list(
      real_mean = mean(df[[atom]]), null_mean = mean(df_null[[atom]]),
      t = as.numeric(t_res$statistic), p = t_res$p.value,
      d = d_res$Cohens_d
    )
  }

  return(list(real_n = nrow(df), null_n = nrow(df_null), atoms = atom_results))
}

# ======================================================================
# 4. CCS VS MMI COMPARISON
# ======================================================================
test_ccs_vs_mmi <- function(df) {
  section("4. CCS VS MMI COMPARISON")

  has_ccs <- "ccs_synergy" %in% names(df) && any(!is.na(df$ccs_synergy))
  if (!has_ccs) {
    cat("\n  WARNING: CCS columns not found. Skipping.\n")
    return(list(skipped = TRUE))
  }

  cat(sprintf("\n  MMI synergy:  mean = %.4f (SD = %.4f)\n", mean(df$synergy), sd(df$synergy)))
  cat(sprintf("  CCS synergy:  mean = %.4f (SD = %.4f)\n", mean(df$ccs_synergy), sd(df$ccs_synergy)))

  cat(sprintf("\n  MMI redundancy:  mean = %.4f (SD = %.4f)\n", mean(df$redundancy), sd(df$redundancy)))
  cat(sprintf("  CCS redundancy:  mean = %.4f (SD = %.4f)\n", mean(df$ccs_redundancy), sd(df$ccs_redundancy)))

  cor_red <- cor.test(df$redundancy, df$ccs_redundancy)
  cat(sprintf("\n  Redundancy correlation (MMI vs CCS): r = %.4f\n", cor_red$estimate))

  t_syn <- t.test(df$synergy, df$ccs_synergy, paired = TRUE)
  cat(sprintf("  Paired t-test (synergy): t(%d) = %.4f, p = %.2e\n",
              t_syn$parameter, t_syn$statistic, t_syn$p.value))

  cat(sprintf("\n  CCS absorbs MMI synergy into redundancy:\n"))
  cat(sprintf("    CCS red - MMI red = %.4f ≈ |MMI synergy| = %.4f\n",
              mean(df$ccs_redundancy) - mean(df$redundancy), abs(mean(df$synergy))))

  return(list(
    mmi_synergy = mean(df$synergy), ccs_synergy = mean(df$ccs_synergy),
    mmi_redundancy = mean(df$redundancy), ccs_redundancy = mean(df$ccs_redundancy),
    red_correlation = as.numeric(cor_red$estimate),
    paired_t = as.numeric(t_syn$statistic), paired_p = t_syn$p.value
  ))
}

# ======================================================================
# 5. SEMANTIC SIMILARITY CORRELATIONS
# ======================================================================
test_similarity_correlations <- function(df) {
  section("5. SEMANTIC SIMILARITY CORRELATIONS")

  cat("\n  Premise-premise similarity:\n")
  cat(sprintf("    Mean = %.4f (SD = %.4f)\n", mean(df$premise_similarity), sd(df$premise_similarity)))

  # Synergy ~ premise similarity
  r_syn <- cor.test(df$synergy, df$premise_similarity)
  cat(sprintf("\n  Synergy ~ Premise similarity: r = %.4f, p = %.4f\n",
              r_syn$estimate, r_syn$p.value))

  # Redundancy ~ premise similarity
  r_red <- cor.test(df$redundancy, df$premise_similarity)
  cat(sprintf("  Redundancy ~ Premise similarity: r = %.4f, p = %.4f\n",
              r_red$estimate, r_red$p.value))

  # Unique info ~ premise-claim similarities
  r_unq1 <- cor.test(df$unique_x1, df$p1_claim_similarity)
  r_unq2 <- cor.test(df$unique_x2, df$p2_claim_similarity)
  cat(sprintf("\n  Unique(P1) ~ P1-Claim similarity: r = %.4f, p = %.4f\n",
              r_unq1$estimate, r_unq1$p.value))
  cat(sprintf("  Unique(P2) ~ P2-Claim similarity: r = %.4f, p = %.4f\n",
              r_unq2$estimate, r_unq2$p.value))

  return(list(
    premise_similarity = list(mean = mean(df$premise_similarity), sd = sd(df$premise_similarity)),
    synergy_premise_sim = list(r = as.numeric(r_syn$estimate), p = r_syn$p.value),
    redundancy_premise_sim = list(r = as.numeric(r_red$estimate), p = r_red$p.value),
    unique_x1_p1_claim_sim = list(r = as.numeric(r_unq1$estimate), p = r_unq1$p.value),
    unique_x2_p2_claim_sim = list(r = as.numeric(r_unq2$estimate), p = r_unq2$p.value)
  ))
}

# ======================================================================
# COMPOSITE FIGURE
# ======================================================================
make_figure <- function(df) {
  section("GENERATING FIGURE")

  # Load null data
  df_null <- NULL
  if (file.exists(NULL_RESULTS_FILE)) {
    df_null <- read_csv(NULL_RESULTS_FILE, show_col_types = FALSE) %>%
      filter(token_count > 0, !is.na(synergy))
  }

  # Shared theme
  theme_paper <- theme_minimal(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(size = 11, face = "bold"),
      plot.tag = element_text(size = 13, face = "bold")
    )

  # ── Panel A: PID atom distributions (violin + box) ──────────────────
  atom_df <- df %>%
    select(redundancy, unique_x1, unique_x2, synergy) %>%
    pivot_longer(everything(), names_to = "atom", values_to = "value") %>%
    mutate(atom = factor(atom,
      levels = c("redundancy", "unique_x1", "unique_x2", "synergy"),
      labels = c("Redundancy", "Unique\n(Premise 1)", "Unique\n(Premise 2)", "Synergy")
    ))

  panel_a <- ggplot(atom_df, aes(x = atom, y = value, fill = atom)) +
    geom_violin(alpha = 0.7, colour = NA, scale = "width") +
    geom_boxplot(width = 0.15, outlier.size = 0.5, fill = "white", alpha = 0.8) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40", linewidth = 0.3) +
    scale_fill_manual(values = c("#64B5F6", "#81C784", "#FFD54F", "#E57373")) +
    labs(x = NULL, y = "Bits per token") +
    theme_paper +
    theme(legend.position = "none")

  # ── Panel B: Real vs Null synergy ──────────────────────────────────
  if (!is.null(df_null)) {
    # Null first so Real is plotted on top
    density_df <- bind_rows(
      tibble(synergy = df_null$synergy, Condition = "Null"),
      tibble(synergy = df$synergy, Condition = "Real")
    ) %>%
      mutate(Condition = factor(Condition, levels = c("Null", "Real")))

    real_mean <- mean(df$synergy)
    null_mean <- mean(df_null$synergy)

    # Red spectrum: Real = synergy color (#E57373), Null = muted red (#B0706F)
    panel_b <- ggplot(density_df, aes(x = synergy, fill = Condition)) +
      geom_density(alpha = 0.6, colour = "grey30", linewidth = 0.3) +
      geom_vline(xintercept = real_mean, linetype = "dashed",
                 colour = "#C62828", linewidth = 0.6) +
      geom_vline(xintercept = null_mean, linetype = "dashed",
                 colour = "#8D6E63", linewidth = 0.6) +
      scale_fill_manual(values = c("Null" = "#B0706F", "Real" = "#E57373")) +
      labs(x = "Synergy (bits per token)", y = "Density", fill = NULL) +
      theme_paper +
      theme(
        legend.position = "right",
        legend.background = element_rect(fill = "white", colour = NA),
        legend.key.size = unit(0.4, "cm"),
        legend.text = element_text(size = 9)
      )
  } else {
    panel_b <- ggplot() + annotate("text", x = 0.5, y = 0.5, label = "Null data not available") +
      theme_void()
  }

  # ── Compose figure (2 panels) ────────────────────────────────────
  fig <- (panel_a | panel_b) +
    plot_annotation(tag_levels = "A") &
    theme(plot.margin = margin(5, 10, 5, 5))

  ggsave(FIGURE_FILE, fig, width = 10, height = 4, dpi = 300)
  cat(sprintf("  Saved: %s\n", FIGURE_FILE))
}

# ======================================================================
# MAIN
# ======================================================================
main <- function() {
  cat(strrep("=", 70), "\n")
  cat("AAE CORPUS — PID ANALYSIS (METHODS PAPER)\n")
  cat(strrep("=", 70), "\n")

  df <- load_data()
  results <- list()

  results$descriptive <- descriptive_statistics(df)
  results$synergy_test <- test_synergy_nonzero(df)
  results$null_model <- test_null_model(df)
  results$ccs_vs_mmi <- test_ccs_vs_mmi(df)
  results$similarity <- test_similarity_correlations(df)

  make_figure(df)

  write_json(results, OUTPUT_FILE, pretty = TRUE, auto_unbox = TRUE)
}

main()
